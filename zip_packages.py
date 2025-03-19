import argparse
import os
import zipfile

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Zip packages")
parser.add_argument("-f", "--folder", type=str, required=True, help="ForAug folder to work in.")
parser.add_argument("-p", "--part", choices=["train", "val"], required=True, help="Part of dataset to zip.")
parser.add_argument("-i", "--images", choices=["foregrounds", "backgrounds"], required=True, help="Images to zip.")

args = parser.parse_args()
ending = "WEBP" if args.images == "foregrounds" else "JPEG"

# gather files
classes = [
    f
    for f in os.listdir(os.path.join(args.folder, args.part, args.images))
    if os.path.isdir(os.path.join(args.folder, args.part, args.images, f))
]
assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"

files = [
    f"{c}/{f}"
    for c in tqdm(classes, desc="gathering files")
    for f in os.listdir(os.path.join(args.folder, args.part, args.images, c))
    if f.endswith(ending)
]

_EXPECTED_FILES = (1274557 if args.images == "foreground" else 1274556) if args.part == "train" else 49751
assert len(files) == _EXPECTED_FILES, f"Expected {_EXPECTED_FILES} files, got {len(files)}"

with zipfile.ZipFile(os.path.join(args.folder, f"{args.images}_{args.part}.zip"), "w") as zf:
    for file in tqdm(files, desc="zipping files"):
        zf.write(os.path.join(args.folder, args.part, args.images, file), file)
