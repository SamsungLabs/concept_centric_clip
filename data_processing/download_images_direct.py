from PIL import Image
import io
from urllib.request import urlopen
import string
import random
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser
import pickle as pkl
import csv
from copy import deepcopy


FILENAME_LENGTH = 8

def get_random_string(length=FILENAME_LENGTH):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_string


def read_tsv(file_path: str|Path):
    rows = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        rows = [row for row in tsv_reader]
    return rows


def download_image(image_url):
    try:
        fd = urlopen(image_url)
        image_file = io.BytesIO(fd.read())
        image = Image.open(image_file).convert("RGB")
    except KeyboardInterrupt:
        raise
    except:
        image = None
    return image


def save_image(image: Image.Image, image_root_dir: str|Path, random_string: str):
    # split the string into 4 parts
    NUM_SPLITS = 4
    n = len(random_string)
    part_n = n // NUM_SPLITS
    # construct directory & file path
    dir_path = Path(image_root_dir)
    for i in range(NUM_SPLITS-1): # only construct dir from the first 3 parts
        dir_path = dir_path / random_string[i*part_n:(i+1)*part_n]
    dir_path.mkdir(exist_ok=True, parents=True)
    file_path = dir_path / f"{random_string}.jpg"
    # save image
    image.save(str(file_path))
    return str(file_path.relative_to(image_root_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf-cache-dir", type=str, default="./data_dir/hf_cache")
    parser.add_argument("--image-dir", type=str, default="./data_dir/images")
    parser.add_argument("--output-pkl", type=str, default=None)
    parser.add_argument("--cc3m-tsv", type=str, default="/home/SERILOCAL/hai.xuanpham/Downloads/Train_GCC-training.tsv")
    
    args = parser.parse_args()

    hf_cache_dir = args.hf_cache_dir
    image_dir = args.image_dir
    Path(image_dir).mkdir(exist_ok=True, parents=True)

    if args.output_pkl is None:
        pkl_file = Path(image_dir) / "cc3m_images.pkl"
    else:
        pkl_file = args.output_pkl

    # 1. read original cc3m metadata
    orginal_cc3m_metadata = read_tsv(args.cc3m_tsv)

    # 2. download DreamLIP data from HF # this include ~30m samples from cc3m, cc12m and yfcc15m, thus we must filter to select on cc3m samples
    dataset = load_dataset("qidouxiong619/dreamlip_long_captions", split="train", cache_dir=hf_cache_dir)

    # 3. Merge data
    original_URLs = set([x[1] for x in orginal_cc3m_metadata])
    valid_indices = []
    for i, row in enumerate(tqdm(dataset, total=len(dataset), desc="URL filtering")):
        if row["Image Path"] in original_URLs:
            valid_indices.append(i)

    # 2. download images
    sampled_names = set()

    image_files = {} # to store the downloaded image paths
    
    for ie, i in enumerate(tqdm(valid_indices, total=len(valid_indices), desc="Downloading")):
        row = dataset[i]
        image = download_image(row["Image Path"])
        if image is None:
            continue
        # sample a unique name
        random_string = get_random_string()
        while random_string in sampled_names:
            random_string = get_random_string()
        sampled_names.add(random_string)
        # save the image
        file_path = save_image(image, image_root_dir=image_dir, random_string=random_string)

        captions = deepcopy(row)
        captions.pop("Image Path")

        image_files[str(i)] = {
            "image_path": file_path,
            "image_url": row["Image Path"],
            "sample_index": i,
            "captions": captions,
        }
        if ie % 100 == 99:
            # save intermediate json
            with open(pkl_file, "wb") as fp:
                pkl.dump(image_files, fp)

    print("*******************************************************************")
    print(f"Downloaded: {len(image_files)} images\nMissing {len(dataset) - len(image_files)} images")

    # save the image list
    with open(pkl_file, "wb") as fp:
        pkl.dump(image_files, fp)


if __name__ == "__main__":
    main()