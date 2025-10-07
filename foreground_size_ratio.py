import argparse
import json
import zipfile
from io import BytesIO

import numpy as np
import PIL
from PIL import Image
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Foreground size ratio")
parser.add_argument("-mode", choices=["train", "val"], default="train", help="Train or val data?")
parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use")
args = parser.parse_args()

# if not os.path.exists("foreground_size_ratios.json"):
train = args.mode == "train"
root = args.dataset

cls_bg_ratios = {}
fg_bg_ratio_map = {}

with zipfile.ZipFile(f"{root}/backgrounds_{'train' if train else 'val'}.zip", "r") as bg_zip, zipfile.ZipFile(
    f"{root}/foregrounds_{'train' if train else 'val'}.zip", "r"
) as fg_zip:
    backgrounds = [f for f in bg_zip.namelist() if f.endswith(".JPEG")]
    foregrounds = [f for f in fg_zip.namelist() if f.endswith(".WEBP")]

    print(f"Bgs: {backgrounds[:5]}, ...\nFgs: {foregrounds[:5]}, ...")

    for bg_name in tqdm(backgrounds):

        fg_name = bg_name.replace("backgrounds", "foregrounds").replace("JPEG", "WEBP")
        if fg_name not in foregrounds:
            tqdm.write(f"Skipping {bg_name} as it has no corresponding foreground")
            fg_bg_ratio_map[bg_name] = 0.0
            continue

        with bg_zip.open(bg_name) as f:
            bg_data = BytesIO(f.read())
            try:
                bg_img = Image.open(bg_data)
            except PIL.UnidentifiedImageError as e:
                print(f"Error with file={bg_name}")
                raise e
            bg_img_size = bg_img.size

        with fg_zip.open(fg_name) as f:
            fg_data = BytesIO(f.read())
            try:
                fg_img = Image.open(fg_data)
            except PIL.UnidentifiedImageError as e:
                print(f"Error with file={fg_name}")
                raise e
            fg_img_size = fg_img.size
            fg_img_pixel_size = int(np.sum(fg_img.split()[-1]) / 255)

        img_cls = bg_name.split("/")[-2]
        if img_cls not in cls_bg_ratios:
            cls_bg_ratios[img_cls] = []

        cls_bg_ratios[img_cls].append(fg_img_pixel_size / (bg_img_size[0] * bg_img_size[1]))
        fg_bg_ratio_map[bg_name] = fg_img_pixel_size / (bg_img_size[0] * bg_img_size[1])

with open(f"{root}/fg_bg_ratios_{args.mode}.json", "w") as f:
    json.dump(fg_bg_ratio_map, f)
print(f"Saved fg_bg_ratios_{args.mode} to disk. Exiting.")
