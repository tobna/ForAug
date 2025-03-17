import argparse
import itertools
import os
import pickle
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def rmse(in_img, fg_img, offsets, fg_mask):
    """Calculate the RMSE between the foreground and extracting a foreground from the in image given the mask and offset."""
    x_offs, y_offs = offsets
    new_fg = in_img[y_offs : y_offs + fg_img.shape[0], x_offs : x_offs + fg_img.shape[1]]
    fg_diff = (fg_img[:, :, :3] / 255 - new_fg / 255) * fg_mask
    diff = np.mean(np.square(fg_diff)) ** 0.5  # RSME
    return diff, x_offs, y_offs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForNet DS -> ImageNet diffs")
    parser.add_argument("-fn", "--fornet", type=str, required=True, help="ForNet/TinyForNet folder")
    parser.add_argument("-in", "--imagenet", type=str, required=True, help="ImageNet folder")
    parser.add_argument("-s", "--split", choices=["val", "train"], required=True, help="Split to process")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument(
        "-id", type=int, default=0, help="Id of this worker process (for splitting into multiple processes)"
    )
    parser.add_argument("-n", "--num_processes", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output folder")

    args = parser.parse_args()
    assert args.id < args.num_processes, f"Id {args.id} is greater than number of processes {args.num_processes}"

    fg_zip = zipfile.ZipFile(os.path.join(args.fornet, f"foregrounds_{args.split}.zip"))
    bg_zip = zipfile.ZipFile(os.path.join(args.fornet, f"backgrounds_{args.split}.zip"))
    foregrounds = set(fg_zip.namelist())
    backgrounds = set(bg_zip.namelist())
    fg_name_start = "/".join(next(iter(foregrounds)).split("/")[:-2])
    bg_name_start = "/".join(next(iter(backgrounds)).split("/")[:-2])
    foregrounds = ["/".join(f.split("/")[-2:]).split(".")[0] for f in foregrounds]
    backgrounds = ["/".join(f.split("/")[-2:]).split(".")[0] for f in backgrounds]

    keys = set(foregrounds).union(set(backgrounds))
    keys = sorted([key for key in keys if "/" in key and key.startswith("n") and not key.endswith("/")])

    if args.num_processes > 1:
        total_keys = len(keys)
        start_idx = args.id * total_keys // args.num_processes
        end_idx = (args.id + 1) * total_keys // args.num_processes
        keys = keys[start_idx:end_idx]

    for idx, key in enumerate(tqdm(keys, disable=args.num_processes > 1)):
        # print(key)
        folder = os.path.join(args.output, args.split, key.split("/")[0])
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(os.path.join(folder, f"{key.split('/')[-1]}.pkl")) and not args.overwrite:
            continue

        in_img = Image.open(os.path.join(args.imagenet, args.split, f"{key}.JPEG")).convert("RGB")

        try:
            fg_img = Image.open(fg_zip.open(f"{fg_name_start}/{key}.WEBP")).convert("RGBA")
        except KeyError:
            fg_img = None

        try:
            bg_img = Image.open(bg_zip.open(f"{bg_name_start}/{key}.JPEG")).convert("RGB")
        except KeyError:
            bg_img = None

        patch = {}
        if bg_img is not None:
            if in_img.size != bg_img.size:
                in_img = in_img.resize(bg_img.size)
            in_img = np.array(in_img)

            bg_img = np.array(bg_img)
            bg_diff = bg_img.astype(np.int64) - in_img.astype(np.int64)
            patch["bg_diff"] = bg_diff
            bg_shape = bg_img.shape
        else:
            max_size = max(in_img.size)
            if max_size > 512:
                goal_size = (round(in_img.size[0] * 512 / max_size), round(in_img.size[1] * 512 / max_size))
                in_img = in_img.resize(goal_size)
            in_img = np.array(in_img)
            bg_shape = in_img.shape

        if fg_img is not None:
            fg_img = np.array(fg_img)

            fg_mask = (fg_img[:, :, 3, np.newaxis] > 0) * 1
            with Pool(args.workers) as p:
                results = p.starmap(
                    rmse,
                    zip(
                        itertools.repeat(in_img),
                        itertools.repeat(fg_img),
                        itertools.product(
                            range(0, bg_shape[1] - fg_img.shape[1] + 1),
                            range(0, bg_shape[0] - fg_img.shape[0] + 1),
                        ),
                        itertools.repeat(fg_mask),
                    ),
                )

            diff, x_off, y_off = min(results, key=lambda x: x[0])
            patch["fg_off"] = (x_off, y_off)
            patch["fg_mask"] = fg_mask
        with open(os.path.join(folder, f"{key.split('/')[-1]}.pkl"), "wb") as f:
            pickle.dump(patch, f)

        if args.num_processes > 1:
            print(f"Processed {key}: {idx + 1}/{len(keys)}", flush=True)
