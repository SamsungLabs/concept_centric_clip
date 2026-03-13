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


def download_images(image_root_dir: str|Path, local_webdataset_dir: str|Path = None) -> dict[str, dict[str, str]]:
    if local_webdataset_dir is None:
        base_url = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{i:04d}.tar"
        urls = [base_url.format(i=i) for i in range(576)]
    else:
        # make sure only the training tar files are stored in this directory
        urls = f"{local_webdataset_dir}/*.tar"
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)

    sampled_names = set()

    metadata_output = {}

    for i, row in enumerate(tqdm(dataset, desc="downloading images")):
        image = row["jpg"]
        url = row["json"]["url"]

        # sample a unique name
        random_string = get_random_string()
        while random_string in sampled_names:
            random_string = get_random_string()
        sampled_names.add(random_string)
        # save the image
        file_path = save_image(image, image_root_dir=image_root_dir, random_string=random_string)

        metadata_output[i] = {
            "image_path": file_path,
            "url": url
        }

    return file_path



def main():
    parser = ArgumentParser()
    parser.add_argument("--hf-cache-dir", type=str, default="./data_dir/hf_cache")
    parser.add_argument("--image-dir", type=str, default="./data_dir/images")
    parser.add_argument("--output-pkl", type=str, default=None)
    parser.add_argument("--cc3m-wds-local-path", type=str, default=None)
    
    args = parser.parse_args()

    hf_cache_dir = args.hf_cache_dir
    image_dir = args.image_dir
    Path(image_dir).mkdir(exist_ok=True, parents=True)

    if args.output_pkl is None:
        pkl_file = Path(image_dir) / "cc3m_images.pkl"
    else:
        pkl_file = args.output_pkl

    # 1. download images
    downloaded_files = download_images(image_root_dir=image_dir, local_webdataset_dir=args.cc3m_wds_local_path)
    print(f"number of images downloaded: {len(downloaded_files)}")
    
    # 2. download DreamLIP data from HF # this include ~30m samples from cc3m, cc12m and yfcc15m, thus we must filter to select on cc3m samples
    dataset = load_dataset("qidouxiong619/dreamlip_long_captions", split="train", cache_dir=hf_cache_dir)

    # 3. Merge data
    url_2_images = {}
    for v in downloaded_files.values():
        url_2_images[v["url"]] = url_2_images["image_path"]

    image_files = {} # to store the downloaded image paths
    
    for ie, i in enumerate(tqdm(dataset, total=len(dataset), desc="Merging data")):
        row = dataset[i]
        url = row["Image Path"]
        if url not in url_2_images:
            continue
        
        captions = deepcopy(row)
        captions.pop("Image Path")

        image_files[str(i)] = {
            "image_path": url_2_images[url],
            "image_url": url,
            "sample_index": i,
            "captions": captions,
        }

    print("*******************************************************************")
    print(f"Downloaded: {len(image_files)} images\nMissing {len(dataset) - len(image_files)} images")

    # save the image list
    with open(pkl_file, "wb") as fp:
        pkl.dump(image_files, fp)


if __name__ == "__main__":
    main()