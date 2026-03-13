import pickle as pkl
import webdataset as wds
from PIL import Image
from pathlib import Path
import math
from datasets import load_dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser
from copy import deepcopy

def pack_data(
        image_root_dir: str,
        metadata_pkl_file: str,
        ##############################
        output_wds_root_dir: str,
        num_samples_per_shard: int = 2000
):
    output_wds_root_dir = Path(output_wds_root_dir)
    output_wds_root_dir.mkdir(exist_ok=True, parents=True)

    # load sample metadata
    with open(metadata_pkl_file, "rb") as fp:
        cc3m_metadata = pkl.load(fp) # dict {"{index}": "{path_str}"}
    valid_data_indices = [int(key) for key in cc3m_metadata.keys()]
    valid_data_indices.sort()

    # start packing
    # get the number of shards (tar files)
    shards_num = int(math.ceil(len(cc3m_metadata)) / num_samples_per_shard)
    tar_pbar = tqdm(total=shards_num, desc="packing TAR", position=1)
    shard_idx = 0

    tar_name = f"{shard_idx:05d}"
    tar_fpath = output_wds_root_dir / f"{tar_name}.tar"
    tar_writer = wds.TarWriter(str(tar_fpath))

    for i, sample_index in enumerate(tqdm(valid_data_indices, desc="sample", position=0)):
        # get metadata
        captions = deepcopy(cc3m_metadata[str(sample_index)]["captions"])
        caption = captions["shortLLA_captions"]
        # add nounphrases to metadata
        nounphrases = cc3m_metadata[str(sample_index)].get("nounphrases", None)
        if nounphrases is not None:
            captions["nounphrases"] = nounphrases
        # load image
        image_path = Path(image_root_dir) / cc3m_metadata[str(sample_index)]["image_path"]
        with open(image_path, "rb") as fp:
            image_data = fp.read()
        # prepare wds record
        key = image_path.stem
        # write
        tar_writer.write({
            "__key__": key,
            "jpg": image_data,
            "txt": caption,
            "json": captions
        })

        if (i+1) % num_samples_per_shard == 0:
            tar_writer.close()
            tar_pbar.update(1)
            # open a new TAR
            shard_idx += 1
            if shard_idx < shards_num:
                tar_name = f"{shard_idx:05d}"
                tar_fpath = output_wds_root_dir / f"{tar_name}.tar"
                tar_writer = wds.TarWriter(str(tar_fpath))
    
    print("***********************************\nAll done!")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-root-dir", type=str, default="./data_dir/images")
    parser.add_argument("--metadata-pkl-file", type=str, default="./data_dir/images/cc3m_images_nounphrases.pkl")
    parser.add_argument("--output-wds-dir", type=str, default="./data_dir/cc3m_np_wds")
    
    args = parser.parse_args()

    pack_data(
        args.image_root_dir,
        args.metadata_pkl_file,
        args.output_wds_dir
    )