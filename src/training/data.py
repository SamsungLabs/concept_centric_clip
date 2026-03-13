import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import random
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from functools import partial
from pathlib import Path
import glob

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def get_image_paths(image_path):
    print(f"Finding all images in {image_path}.")
    image_extensions = ('png', 'jpg', 'jpeg', 'gif', 'bmp')
    return [os.path.join(root, filename)
            for root, dirs, files in os.walk(image_path, followlinks=True)
            for filename in files
            if filename.lower().endswith(image_extensions)]


class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, images_dir_path=None, num_captions=1,
                distill_tokenizer=None, distill_preprocess=None, **kwargs):

        logging.debug(f'Loading JSON data from {input_filename}.')
        with open(input_filename, 'r') as f:
            data = json.load(f)

        print(f"Found {len(data)} images in the dataset.")

        images = list(data.keys())
        self.captions = [data[img_key] for img_key in images]
        num_caps = max([len(i) for i in self.captions[:100]])

        assert num_captions <= num_caps, f"Only {num_caps} were generate, but requested {num_captions} for training."
        assert images_dir_path is not None, "Please specify images_dir_path"

        print(f"Using {num_captions} captions per image. Found {num_caps} per image in the dataset.")

        image_paths = get_image_paths(images_dir_path)
        image_keys = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
        image_keys_to_paths = dict(zip(image_keys, image_paths))
        self.images = [image_keys_to_paths[key] for key in images]

        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.num_captions = num_captions

        self.distill_tokenizer = distill_tokenizer
        self.distill_preprocess = distill_preprocess

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])))
        image_captions = self.captions[idx]

        if self.num_captions == 1:
            captions = image_captions[0]
            texts = self.tokenize([str(captions)])[0]

            if self.distill_preprocess is not None:
                image_distill = self.distill_preprocess(Image.open(str(self.images[idx])))
                texts_distill = self.distill_tokenizer([str(captions)])[0]
                return image, image_distill, texts, texts_distill

            return image, texts

        if self.num_captions > len(image_captions):
            texts = random.choices(image_captions, k=self.num_captions)
        else:
            texts = image_captions[:self.num_captions]

        texts_tokenized = self.tokenize([str(txt) for txt in texts])

        if self.distill_preprocess is not None:
            image_distill = self.distill_preprocess(Image.open(str(self.images[idx])))
            texts_distill = self.distill_tokenizer([str(txt) for txt in texts])
            return image, image_distill, texts_tokenized, texts_distill
        
        return image, texts_tokenized


def get_json_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None,
                    distill_tokenizer=None, distill_preprocess=None, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    dataset = JsonDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        num_captions=args.num_captions,
        images_dir_path=args.images_dir_path,
        distill_tokenizer=distill_tokenizer,
        distill_preprocess=distill_preprocess,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                            tokenizer=None, distill_tokenizer=None, distill_preprocess=None, **kwargs):

        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        
        self.distill_tokenizer = distill_tokenizer
        self.distill_preprocess = distill_preprocess

        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts_tokenized = self.tokenize([str(self.captions[idx])])[0]

        if self.distill_preprocess is not None:
            images_distil = self.distill_preprocess(Image.open(str(self.images[idx])))
            texts_distill = self.distill_tokenizer([str(self.captions[idx])])[0]
            return images, images_distil, texts_tokenized, texts_distill

        return images, texts_tokenized


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if len(filesample) == 0 or "fname" not in filesample or "data" not in filesample:
            #print(filesample)
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def tokenize_text(txt, tokenizer, num_captions=1):
    if num_captions > 1:
        captions = txt.split(";;;")
        return tokenizer(captions)

    return tokenizer(txt)[0]


def repack_data(sample, num_captions=1, captions_key='blip2_captions'):
    sample['jpg'] = sample['image'] if 'image' in sample else sample['jpg']
    json_data = json.loads(sample['json'])
    captions = json_data[captions_key]

    if num_captions > len(captions):
        captions = random.choices(captions, k=num_captions)
    else:
        captions = random.sample(captions, num_captions)

    captions = ";;;".join(captions)
    sample['txt'] =  str.encode(captions)

    if 'image' in sample:
        del sample['image']
    if 'json' in sample:
        del sample['json']
    return sample



def duplicate(sample, is_distill=False):
    if is_distill:
        sample['text_distill'] = sample['text']
        sample['image_distill'] = sample['image']
    return sample

def get_wds_dataset(
        args,
        preprocess_img,
        is_train,
        epoch=0,
        floor=False,
        tokenizer=None,
        distill_tokenizer=None,
        distill_preprocess=None,
        **kwargs,
    ):

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    
    is_distill = (distill_tokenizer is not None)
    duplicate_fn = partial(duplicate, is_distill=is_distill)

    text_processor = partial(tokenize_text, tokenizer=tokenizer, num_captions=args.num_captions)
    if is_distill:
        distill_text_processor = partial(tokenize_text, tokenizer=distill_tokenizer, num_captions=args.num_captions)

        preprocess_step = wds.map_dict(
            image=preprocess_img, 
            image_distill=distill_preprocess, 
            text=text_processor, 
            text_distill=distill_text_processor
        )
    else:
        preprocess_step = wds.map_dict(
            image=preprocess_img, 
            text=text_processor
        )

    if is_distill:
        tuple_step = wds.to_tuple("image", "image_distill", "text", "text_distill")
    else:
        tuple_step = wds.to_tuple("image", "text")

    pipeline.extend([
        wds.map(partial(repack_data, num_captions=args.num_captions, captions_key=args.captions_key)),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", wds.handle_extension("txt", lambda x: x.decode('utf-8')), handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map(duplicate_fn),
        preprocess_step,
        tuple_step,
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        prefetch_factor=4,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, distill_tokenizer=None, distill_preprocess=None, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        distill_tokenizer=distill_tokenizer,
        distill_preprocess=distill_preprocess,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
            **kwargs,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

######################################################################################
class NpyDataset(Dataset):
    def __init__(self,samples_path, transforms=None, train_num_samples=None, tokenizer=None, split=None):
        if split==None:
            if 'val' in samples_path:
                self.split = 'val'
            else:
                self.split = 'train'
        else:
            self.split = split

        if 'coco' in samples_path:
            self.data = 'coco'
        else:
            self.data = 'cc3m'

        if os.path.isdir(samples_path):
            # load and merge all splited files in the dir
            data_file_splits = glob.glob(os.path.join(samples_path,'*.npy'))
            print(f'merging {len(data_file_splits)} splied files from {samples_path}')
            self.samples=[]
            for file_split in data_file_splits:
                self.samples.extend(self.loadList(file_split))
        else:
            # load single splited file fiven the path
            self.samples = self.loadList(samples_path)

        if train_num_samples:
            self.samples = self.samples[:train_num_samples]
        self.transforms = transforms
        self.tokenize = tokenizer


    def loadList(self, file_path):
        # the filename should mention the extension '.npy'
        tempNumpyArray = np.load(file_path, allow_pickle=True)
        return tempNumpyArray.tolist()


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,index):
        captions = torch.stack([self.tokenize([str(self.samples[index]['caption'])])[0],
                                self.tokenize([str(self.samples[index]['relation_aug_caption'])])[0],
                                self.tokenize([str(self.samples[index]['adj_aug_caption'])])[0],
                                self.tokenize([str(self.samples[index]['noun_aug_caption'])])[0],
                                self.tokenize([str(self.samples[index]['verb_aug_caption'])])[0]])
        if self.data=='coco':
            image_id = self.samples[index]['image_id']
            data_split = 'train2014' if self.split=='train' else "val2014"
            image_path = os.path.join(COCO_DATASET_ROOT,data_split,f"COCO_{data_split}_{'0'*(12-len(str(image_id)))}{image_id}.jpg")
            image = self.transforms(Image.open(image_path).convert("RGB"))
        else:
            image = self.transforms(Image.open(str(self.samples[index]['image_path'])).convert("RGB"))
        
        valid_caption_mask=torch.tensor(self.samples[index]['valid_caption'])
        
        return image, captions, valid_caption_mask


class HardNegative_Collate:
    def __call__(self, batch):
        img = torch.stack([example[0] for example in batch])
        true_caption = torch.stack([example[1][0] for example in batch])
        hard_negative = torch.cat([example[1][1:] for example in batch])
        text = torch.cat([true_caption, hard_negative])
        valid_caption_mask = torch.stack([example[2] for example in batch])
        return img, text, valid_caption_mask
    

def get_npy_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = NpyDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        train_num_samples=args.train_num_samples

    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    collate=HardNegative_Collate()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

##################################################################################
class CC3m_Caption_Loader():
    def __init__(self, tokenizer, caption_path: str, start_token: int=None, end_token: int=None, 
                 extra_da: bool = True,
                 return_token_mask_caption=False, 
                 return_token_mask_negative=False,
                 return_nounphrases=False):
        self.tokenize = tokenizer
        import pickle
        with open(caption_path, "rb") as fp:
            self.captions = pickle.load(fp)
        
        self.start_token = start_token
        self.end_token = end_token
        self.return_token_mask_caption = return_token_mask_caption
        self.return_token_mask_negative = return_token_mask_negative
        self.return_nounphrases = return_nounphrases
        self.extra_da = extra_da

    
    def get_valid_token_mask(self, captions: torch.Tensor) -> torch.Tensor:
        if self.end_token is None:
            # return equal lengths of MAX_TOKENS (default = 77)
            return torch.ones_like(captions, type=torch.long)
        
        if self.start_token is not None and self.end_token is not None:
            # openai tokenizer
            mask = torch.where(captions > 0, 1, 0)
            mask = torch.where(captions == self.start_token, 0, mask)
            mask = torch.where(captions == self.end_token, 0, mask)
        elif self.start_token is None and self.end_token is not None:
            # siglip T5-based tokenizer
            # end_token == 1
            mask = torch.where(captions != self.end_token, 1, 0)
        else:
            raise ValueError("the tokenizer format is not supported")

        return mask # (num_captions, max_caption_length)

    
    def get_item_simple(self, key: str):
        sample = self.captions[key]

        
        caption = self.tokenize(str(sample['caption']))[0],
                                
        return caption

    
    def get_item_da(self, key: str):
        sample = self.captions[key]

        
        valid_caption_mask = [1] * 5
        for i, k in enumerate(['caption', 'relation_aug_caption', 'adj_aug_caption', 'noun_aug_caption', 'verb_aug_caption']):
            if sample[k] == "###":
                valid_caption_mask[i] = 0

        captions = torch.stack([self.tokenize([str(sample['caption'])])[0],
                                self.tokenize([str(sample['relation_aug_caption'])])[0],
                                self.tokenize([str(sample['adj_aug_caption'])])[0],
                                self.tokenize([str(sample['noun_aug_caption'])])[0],
                                self.tokenize([str(sample['verb_aug_caption'])])[0]])
        valid_caption_mask = torch.tensor(valid_caption_mask) #torch.tensor(sample['valid_caption'])
        # except Exception as error:
        #     traceback.print_exc()
        #     print("Sample: ", sample)
        #     raise RuntimeError

        # get the lengths -> token_mask
        if self.return_token_mask_caption and self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions)
        elif self.return_token_mask_caption:
            token_mask = self.get_valid_token_mask(captions[0:1, :])
        elif self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions[1:, :])
        else:
            token_mask = None

        # get nounphrases if available
        noun_phrases = None
        if self.return_nounphrases and "nounphrases" in sample:
            nps = sample['nounphrases']
            if len(nps) == 0:
                nps = [str(sample["caption"])] # so training does not fail when there's no nounphrase
            noun_phrases = self.tokenize(nps)
            noun_phrases_mask = self.get_valid_token_mask(noun_phrases)
        
        # Done. Returning
        if token_mask is not None and noun_phrases is not None:
            ret = (captions, valid_caption_mask, token_mask, noun_phrases, noun_phrases_mask)
        elif token_mask is not None:
            ret = (captions, valid_caption_mask, token_mask)
        elif noun_phrases is not None:
            ret = (captions, valid_caption_mask, noun_phrases, noun_phrases_mask)
        else:
            ret = (captions, valid_caption_mask)

        return ret
    
    
    def __call__(self, key: str):
        if self.extra_da:
            return self.get_item_da(key)
        else:
            return self.get_item_simple(key)


# special data collate function for custom CC3m dataset
def simple_collate_fn(batch):
    img = torch.stack([example[0] for example in batch])
    caption = torch.stack([example[1][0] for example in batch])
    return [img, caption]


class CC3m_custom_collate_fn():
    def __init__(self, return_token_mask=False, return_nounphrases=False, return_negative_nounphrases=False):
        self.return_token_mask = return_token_mask
        self.return_nounphrases = return_nounphrases
        self.return_negative_nounphrases = return_negative_nounphrases


    def __call__(self, batch):
        """
        Returning a tuple X
        - Fixed outputs:
            X[0]: img
            X[1]: text
            X[2]: valid_caption_mask
        - Variable outputs:
            - If return_token_mask == True and return_nounphrases == True
                X[3] = token_mask
                X[4] = noun_phrases
                X[5] = noun_phrases_sample_indices
            - If return_token_mask == False and return_nounphrases == True
                X[3] = noun_phrases
                X[4] = noun_phrases_sample_indices
            - If return_token_mask == True and return_nounphrases == False
                X[3] = token_mask
        """
        # for CE-CLIP data
        img = torch.stack([example[0] for example in batch])
        true_caption = torch.stack([example[1][0][0] for example in batch])
        hard_negative = torch.cat([example[1][0][1:] for example in batch])
        text = torch.cat([true_caption, hard_negative])
        valid_caption_mask = torch.stack([example[1][1] for example in batch])
        
        ret = [img, text, valid_caption_mask]

        # check for data structure
        if self.return_nounphrases and self.return_negative_nounphrases and self.return_token_mask and len(batch[0][1]) != 7:
            raise ValueError(f"The batch does not contain correct data. Expecting 7 item, received {len(batch[0][1])} items.")
        if self.return_nounphrases and self.return_negative_nounphrases and not self.return_token_mask and len(batch[0][1]) != 6:
            raise ValueError(f"The batch does not contain correct data. Expecting 6 item, received {len(batch[0][1])} items.")
        if self.return_nounphrases and self.return_token_mask and not self.return_negative_nounphrases and len(batch[0][1]) != 5:
            raise ValueError(f"The batch does not contain correct data. Expecting 5 item, received {len(batch[0][1])} items.")
        if self.return_nounphrases and not self.return_token_mask and not self.return_negative_nounphrases and len(batch[0][1]) != 4:
            raise ValueError(f"The batch does not contain correct data. Expecting 4 item, received {len(batch[0][1])} items.")
        if not self.return_nounphrases and self.return_token_mask and len(batch[0][1]) != 3:
            raise ValueError(f"The batch does not contain correct data. Expecting 3 item, received {len(batch[0][1])} items.")
        
        # get token_mask (for text lengths)
        if self.return_token_mask:
            # there's valid token mask
            if batch[0][1][2].shape[0] == 1 or batch[0][1][2].shape[0] == 4: # only true captions or negative captions
                token_mask = torch.cat([example[1][2] for example in batch])
            else:
                true_caption_token_mask = torch.stack([example[1][2][0] for example in batch])
                negative_caption_token_mask = torch.cat([example[1][2][1:] for example in batch])
                token_mask = torch.cat((true_caption_token_mask, negative_caption_token_mask), dim=0)
            ret.append(token_mask)
        
        # get nounphrases
        if self.return_nounphrases:
            if not self.return_token_mask:
                nounphrases = [example[1][2] for example in batch]
                nounphrases_token_mask = [example[1][3] for example in batch]
            else:
                nounphrases = [example[1][3] for example in batch]
                nounphrases_token_mask = [example[1][4] for example in batch]
            sample_indices = []
            for i, nps in enumerate(nounphrases):
                sample_indices.extend([i]*len(nps))
            sample_indices = torch.tensor(sample_indices)
            nounphrases = torch.cat(nounphrases, dim=0)
            nounphrases_token_mask = torch.cat(nounphrases_token_mask, dim=0)
            ret.append(nounphrases)
            ret.append(nounphrases_token_mask)
            ret.append(sample_indices)

        if self.return_negative_nounphrases:
            if not self.return_token_mask:
                hn_nounphrases = [example[1][4] for example in batch]
                hn_nounphrases_token_mask = [example[1][5] for example in batch]
            else:
                hn_nounphrases = [example[1][5] for example in batch]
                hn_nounphrases_token_mask = [example[1][6] for example in batch]
            ret_hn_sample_indices = []
            ret_hn_nps = []
            ret_hn_np_token_mask = []
            for i, nps in enumerate(hn_nounphrases):
                if isinstance(nps, list) and len(nps) == 0:
                    continue
                ret_hn_nps.append(nps)
                ret_hn_np_token_mask.append(hn_nounphrases_token_mask[i])
                ret_hn_sample_indices.extend([i]*len(nps))
            ret_hn_sample_indices = torch.tensor(ret_hn_sample_indices)
            hn_nounphrases = torch.cat(ret_hn_nps, dim=0)
            hn_nounphrases_token_mask = torch.cat(ret_hn_np_token_mask, dim=0)
            ret.append(hn_nounphrases)
            ret.append(hn_nounphrases_token_mask)
            ret.append(ret_hn_sample_indices)
        
        return tuple(ret)


def get_cc3m_custom_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, **kwargs):
    assert args.cc3m_captions is not None and Path(args.cc3m_captions).exists()
    #print("Load CC3m custom dataset")

    start_token = None
    end_token = None
    
    if args.model == "ViT-B-32" and args.pretrained == "openai":
        start_token = 49406
        end_token = 49407
    elif args.model == "ViT-B-16-SigLIP": # and args.pretrained == "webli":
        end_token = 1
    else:
        if args.output_tokens:
            raise RuntimeError(f"{args.model} from {args.pretrained} do not support returning raw token features")
    
    return_token_mask_caption = args.scan_loss
    return_nounphrases = args.np_loss
    complex_loader_mode = args.extra_da or args.np_loss or args.output_tokens

    caption_loader = CC3m_Caption_Loader(tokenizer=tokenizer, 
                                         caption_path=args.cc3m_captions,
                                         start_token=start_token,
                                         end_token=end_token,
                                         return_token_mask_caption=return_token_mask_caption,
                                         return_token_mask_negative=return_token_mask_caption,
                                         return_nounphrases=return_nounphrases,
                                         extra_da=complex_loader_mode)

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="__key__"),
        wds.map_dict(image=preprocess_img, text=caption_loader),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train, 
                    collation_fn=CC3m_custom_collate_fn(return_token_mask=return_token_mask_caption, 
                                                        return_nounphrases=return_nounphrases)
                    ) if complex_loader_mode else wds.batched(args.batch_size, partial=not is_train, collation_fn=simple_collate_fn),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

######################################################################################
"""
New custom data loader for newly packed cc30m dataset
"""
class CC30mJsonProcessor():
    def __init__(self, tokenizer, start_token: int=None, end_token: int=None, 
                 extra_da: bool = True,
                 return_token_mask_caption=False, 
                 return_token_mask_negative=False,
                 return_nounphrases=False):
        self.tokenize = tokenizer
        
        self.start_token = start_token
        self.end_token = end_token
        self.return_token_mask_caption = return_token_mask_caption
        self.return_token_mask_negative = return_token_mask_negative
        self.return_nounphrases = return_nounphrases
        self.extra_da = extra_da
    

    def get_valid_token_mask(self, captions: torch.Tensor) -> torch.Tensor:
        if self.end_token is None:
            # return equal lengths of MAX_TOKENS (default = 77)
            return torch.ones_like(captions, type=torch.long)
        
        if self.start_token is not None and self.end_token is not None:
            # openai tokenizer
            mask = torch.where(captions > 0, 1, 0)
            mask = torch.where(captions == self.start_token, 0, mask)
            mask = torch.where(captions == self.end_token, 0, mask)
        elif self.start_token is None and self.end_token is not None:
            # siglip T5-based tokenizer
            # end_token == 1
            mask = torch.where(captions != self.end_token, 1, 0)
        else:
            raise ValueError("the tokenizer format is not supported")

        return mask # (num_captions, max_caption_length)
    

    def get_item_simple(self, json_data):
        caption = str(json_data["short_caption"])
        caption = self.tokenize(caption)[0],     
        return caption
    

    def get_item_da(self, json_data):
        captions = torch.stack([self.tokenize([str(json_data['short_caption'])])[0],
                                self.tokenize([str(json_data['relation_aug_caption'])])[0],
                                self.tokenize([str(json_data['adj_aug_caption'])])[0],
                                self.tokenize([str(json_data['noun_aug_caption'])])[0],
                                self.tokenize([str(json_data['verb_aug_caption'])])[0]])
        valid_caption_mask = torch.tensor(json_data["valid_caption"])

        if self.return_token_mask_caption and self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions)
        elif self.return_token_mask_caption:
            token_mask = self.get_valid_token_mask(captions[0:1, :])
        elif self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions[1:, :])
        else:
            token_mask = None

        # get nounphrases if available
        noun_phrases = None
        if self.return_nounphrases and "nounphrases" in json_data:
            nps = json_data['nounphrases']
            if len(nps) == 0:
                nps = [str(json_data["short_caption"])] # so training does not fail when there's no nounphrase
            noun_phrases = self.tokenize(nps)
            noun_phrases_mask = self.get_valid_token_mask(noun_phrases)
        
        # Done. Returning
        if token_mask is not None and noun_phrases is not None:
            ret = (captions, valid_caption_mask, token_mask, noun_phrases, noun_phrases_mask)
        elif token_mask is not None:
            ret = (captions, valid_caption_mask, token_mask)
        elif noun_phrases is not None:
            ret = (captions, valid_caption_mask, noun_phrases, noun_phrases_mask)
        else:
            ret = (captions, valid_caption_mask)

        return ret
    

    def __call__(self, json_data):
        if self.extra_da:
            return self.get_item_da(json_data)
        else:
            return self.get_item_simple(json_data)


def get_cc30m_custom_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, **kwargs):
    start_token = None
    end_token = None
    
    if args.model == "ViT-B-32" and args.pretrained == "openai":
        start_token = 49406
        end_token = 49407
    elif args.model == "ViT-B-16-SigLIP": # and args.pretrained == "webli":
        end_token = 1
    else:
        if args.output_tokens:
            raise RuntimeError(f"{args.model} from {args.pretrained} do not support returning raw token features")
    
    return_token_mask_caption = args.scan_loss
    return_nounphrases = args.np_loss
    complex_loader_mode = args.extra_da or args.np_loss or args.output_tokens

    caption_loader = CC30mJsonProcessor(tokenizer=tokenizer, 
                                        start_token=start_token,
                                        end_token=end_token,
                                        return_token_mask_caption=return_token_mask_caption,
                                        return_token_mask_negative=return_token_mask_caption,
                                        return_nounphrases=return_nounphrases,
                                        extra_da=complex_loader_mode)

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="json"),
        wds.map_dict(image=preprocess_img, text=caption_loader),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train, 
                    collation_fn=CC3m_custom_collate_fn(return_token_mask=return_token_mask_caption, 
                                                        return_nounphrases=return_nounphrases)
                    ) if complex_loader_mode else wds.batched(args.batch_size, partial=not is_train, collation_fn=simple_collate_fn),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


######################################################################################
"""
New custom data loader for newly packed cc30m dataset - with NEGATIVE NOUNPHRASES
"""
class CC30mJsonProcessor_hn_np():
    def __init__(self, tokenizer, start_token: int=None, end_token: int=None, 
                 extra_da: bool = True,
                 return_token_mask_caption=False, 
                 return_token_mask_negative=False,
                 return_nounphrases=False,
                 return_negative_nounphrases=False,
                 hn_np_balance=False,
                 use_original_caption=False):
        self.tokenize = tokenizer
        
        self.start_token = start_token
        self.end_token = end_token
        self.return_token_mask_caption = return_token_mask_caption
        self.return_token_mask_negative = return_token_mask_negative
        self.return_nounphrases = return_nounphrases
        self.return_negative_nounphrases = return_negative_nounphrases
        self.extra_da = extra_da
        self.hn_np_balance = hn_np_balance
        self.use_original_caption = use_original_caption
    

    def get_valid_token_mask(self, captions: torch.Tensor) -> torch.Tensor:
        if self.end_token is None:
            # return equal lengths of MAX_TOKENS (default = 77)
            return torch.ones_like(captions, type=torch.long)
        
        if self.start_token is not None and self.end_token is not None:
            # openai tokenizer
            mask = torch.where(captions > 0, 1, 0)
            mask = torch.where(captions == self.start_token, 0, mask)
            mask = torch.where(captions == self.end_token, 0, mask)
        elif self.start_token is None and self.end_token is not None:
            # siglip T5-based tokenizer
            # end_token == 1
            mask = torch.where(captions != self.end_token, 1, 0)
        else:
            raise ValueError("the tokenizer format is not supported")

        return mask # (num_captions, max_caption_length)
    

    def get_item_simple(self, json_data):
        if self.use_original_caption:
            caption = str(json_data["caption"])
        else:
            caption = str(json_data["short_caption"])
        caption = self.tokenize(caption)[0],     
        return caption
    
    def _process_hn_nounphrases(self, hn_nounphrases):
        def _is_valid(np):#
            toks = np.lower().split()
            if len(toks) == 1:
                # single word
                return False
            
            if len(toks) == 2 and toks[0] in ["a", "an", "the", "their", "his", "her", "my", "your", "our", "its"]:
                # "a person", "his dog" ...
                return False
            
            return True
        
        num_swapped = len(hn_nounphrases["swapped"])
        num_masked = len(hn_nounphrases["masked"])
        
        if self.hn_np_balance and num_swapped > 0 and num_masked > 0 and num_masked != num_swapped:
            min_num = min(num_swapped, num_masked)
            swapped_nps = random.sample(hn_nounphrases["swapped"], k=min_num)
            masked_nps = random.sample(hn_nounphrases["masked"], k=min_num)
            hn_nps = swapped_nps + masked_nps
        else:
            hn_nps = hn_nounphrases["swapped"] + hn_nounphrases["masked"]

        ret = [np for np in hn_nps if _is_valid(np)]
        
        return ret

    def get_item_da(self, json_data):
        captions = torch.stack([self.tokenize([str(json_data['short_caption'])])[0],
                                self.tokenize([str(json_data['relation_aug_caption'])])[0],
                                self.tokenize([str(json_data['adj_aug_caption'])])[0],
                                self.tokenize([str(json_data['noun_aug_caption'])])[0],
                                self.tokenize([str(json_data['verb_aug_caption'])])[0]])
        valid_caption_mask = torch.tensor(json_data["valid_caption"])

        if self.return_token_mask_caption and self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions)
        elif self.return_token_mask_caption:
            token_mask = self.get_valid_token_mask(captions[0:1, :])
        elif self.return_token_mask_negative:
            token_mask = self.get_valid_token_mask(captions[1:, :])
        else:
            token_mask = None

        # get nounphrases if available
        noun_phrases = None
        if self.return_nounphrases and "nounphrases" in json_data:
            nps = json_data['nounphrases']
            if len(nps) == 0:
                nps = [str(json_data["short_caption"])] # so training does not fail when there's no nounphrase
            noun_phrases = self.tokenize(nps)
            noun_phrases_mask = self.get_valid_token_mask(noun_phrases)

        negative_nounphrases = None
        if self.return_negative_nounphrases and "hn_nounphrases" in json_data:
            hn_nps = self._process_hn_nounphrases(json_data['hn_nounphrases'])
            if len(hn_nps) == 0:
                # pick shortest negative caption from DA data
                min_len = 10000
                hn_cap = None
                for key in ["relation_aug_caption", "adj_aug_caption", "noun_aug_caption", "verb_aug_caption"]:
                    if len(str(json_data[key])) < min_len and str(json_data[key]) != "###":
                        hn_cap = str(json_data[key])
                        min_len = len(str(json_data[key]))
                if hn_cap is None:
                    hn_nps = None
                else:
                    hn_nps = [hn_cap] # so training does not fail when there's no nounphrase
            if hn_nps is not None:
                negative_nounphrases = self.tokenize(hn_nps)
                negative_nounphrases_mask = self.get_valid_token_mask(negative_nounphrases)
            else:
                negative_nounphrases = []
                negative_nounphrases_mask = []

        
        # Done. Returning
        if token_mask is not None and noun_phrases is not None and negative_nounphrases is not None:
            ret = (captions, valid_caption_mask, token_mask, noun_phrases, noun_phrases_mask, negative_nounphrases, negative_nounphrases_mask)
        elif token_mask is None and noun_phrases is not None and negative_nounphrases is not None:
            ret = (captions, valid_caption_mask, noun_phrases, noun_phrases_mask, negative_nounphrases, negative_nounphrases_mask)
        elif token_mask is not None and noun_phrases is not None:
            ret = (captions, valid_caption_mask, token_mask, noun_phrases, noun_phrases_mask)
        elif token_mask is not None and noun_phrases is None:
            ret = (captions, valid_caption_mask, token_mask)
        elif token_mask is None and noun_phrases is not None:
            ret = (captions, valid_caption_mask, noun_phrases, noun_phrases_mask)
        else:
            ret = (captions, valid_caption_mask)

        return ret
    

    def __call__(self, json_data):
        if self.extra_da:
            return self.get_item_da(json_data)
        else:
            return self.get_item_simple(json_data)



def get_cc30m_hn_np_custom_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, **kwargs):
    start_token = None
    end_token = None
    
    if args.model == "ViT-B-32" and args.pretrained == "openai":
        start_token = 49406
        end_token = 49407
    elif args.model == "ViT-B-16-SigLIP": # and args.pretrained == "webli":
        end_token = 1
    else:
        if args.output_tokens:
            raise RuntimeError(f"{args.model} from {args.pretrained} do not support returning raw token features")
    
    return_token_mask_caption = args.scan_loss
    return_nounphrases = args.np_loss
    return_negative_nounphrases = args.np_hard_negative_loss or args.np_hard_negative_flair_loss
    complex_loader_mode = args.extra_da or args.np_loss or args.output_tokens

    caption_loader = CC30mJsonProcessor_hn_np(tokenizer=tokenizer,
                                        start_token=start_token,
                                        end_token=end_token,
                                        return_token_mask_caption=return_token_mask_caption,
                                        return_token_mask_negative=return_token_mask_caption,
                                        return_nounphrases=return_nounphrases,
                                        return_negative_nounphrases=return_negative_nounphrases,
                                        extra_da=complex_loader_mode,
                                        hn_np_balance=args.hn_np_balance,
                                        use_original_caption=args.use_original_caption)

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="json"),
        wds.map_dict(image=preprocess_img, text=caption_loader),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train, 
                    collation_fn=CC3m_custom_collate_fn(return_token_mask=return_token_mask_caption, 
                                                        return_nounphrases=return_nounphrases,
                                                        return_negative_nounphrases=return_negative_nounphrases)
                    ) if complex_loader_mode else wds.batched(args.batch_size, partial=not is_train, collation_fn=simple_collate_fn),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


######################################################################################


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "npy":
        return get_npy_dataset
    elif dataset_type == "cc3m_custom":
        return get_cc3m_custom_dataset
    elif dataset_type == "cc30m_custom":
        return get_cc30m_custom_dataset
    elif dataset_type == "cc30m_custom_hn_np":
        return get_cc30m_hn_np_custom_dataset
    elif dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None, distill_tokenizer=None, distill_preprocess=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True,
                                    epoch=epoch,
                                    tokenizer=tokenizer,
                                    distill_tokenizer=distill_tokenizer,
                                    distill_preprocess=distill_preprocess,
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data