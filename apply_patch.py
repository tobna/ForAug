import argparse
import os
import pickle
import zipfile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="ImageNet Patches -> ForNet")
parser.add_argument("-p", "--patch", type=str, required=True, help="Patch file")
parser.add_argument("-in", "--imagenet", type=str, required=True, help="ImageNet folder")
parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")

args = parser.parse_args()

part = args.patch.split("/")[-1].split(".")[0]

patch_file = zipfile.ZipFile(args.patch)
patches = set(patch_file.namelist())
for p_ in patches:
    if p_.endswith(".pkl"):
        ex_name = p_
        break
patches = sorted(["/".join(pf.split("/")[-2:]).split(".")[0] for pf in patches if "/" in pf and pf.endswith(".pkl")])
patch_name_start = "/".join(ex_name.split("/")[:-2])

for patch in tqdm(patches):
    with patch_file.open(f"{patch_name_start}/{patch}.pkl", "r") as pklf:
        patch_data = pickle.load(pklf)

    in_img = Image.open(os.path.join(args.imagenet, part, f"{patch}.JPEG")).convert("RGB")
    in_img = np.array(in_img)

    os.makedirs(os.path.join(args.output, part, "backgrounds", patch.split("/")[0]), exist_ok=True)
    os.makedirs(os.path.join(args.output, part, "foregrounds", patch.split("/")[0]), exist_ok=True)

    if "bg_diff" in patch_data:
        bg_diff = patch_data["bg_diff"]
        bg_img = in_img.astype(np.int64) + bg_diff
        Image.fromarray(bg_img.clip(0, 255).astype(np.uint8)).save(
            os.path.join(args.output, part, "backgrounds", patch + ".JPEG")
        )

    if "fg_mask" in patch_data:
        x_offs, y_offs = patch_data["fg_off"]
        fg_mask = patch_data["fg_mask"]

        fg_crop = in_img[y_offs : y_offs + fg_mask.shape[0], x_offs : x_offs + fg_mask.shape[1]]
        fg_img = np.concatenate([fg_crop, fg_mask * 255], axis=-1).clip(0, 255).astype(np.uint8)
        Image.fromarray(fg_img).save(os.path.join(args.output, part, "foregrounds", patch + ".WEBP"))
