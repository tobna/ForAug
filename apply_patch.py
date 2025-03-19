import argparse
import gzip
import itertools
import multiprocessing
import os
import pickle
import zipfile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def _extract(args):
    # zip_path, patch_name, outpath, name_start, file_ending, in_path, part = *args
    zip_path = args[0]
    patch_name = args[1]
    outpath = args[2]
    name_start = args[3]
    file_ending = args[4]
    in_path = args[5]
    part = args[6]

    if os.path.exists(os.path.join(outpath, part, "foregrounds", patch_name + ".WEBP")) and os.path.exists(
        os.path.join(outpath, part, "backgrounds", patch_name + ".JPEG")
    ):
        return

    with zipfile.ZipFile(zip_path, "r") as patch_file, (
        patch_file.open(f"{name_start}{patch_name}.{file_ending}", "r")
        if ending == "pkl"
        else gzip.GzipFile(fileobj=patch_file.open(f"{name_start}{patch_name}.{file_ending}", "r"), mode="r")
    ) as pklf:
        patch_data = pickle.load(pklf)

    in_img = Image.open(os.path.join(in_path, part, f"{patch_name}.JPEG")).convert("RGB")

    if "bg_diff" in patch_data:
        if in_img.size != (patch_data["bg_diff"].shape[1], patch_data["bg_diff"].shape[0]):
            in_img = in_img.resize((patch_data["bg_diff"].shape[1], patch_data["bg_diff"].shape[0]))
    else:
        max_size = max(in_img.size)
        if max_size > 512:
            goal_size = (round(in_img.size[0] * 512 / max_size), round(in_img.size[1] * 512 / max_size))
            in_img = in_img.resize(goal_size)

    in_img = np.array(in_img)

    os.makedirs(os.path.join(outpath, part, "backgrounds", patch_name.split("/")[0]), exist_ok=True)
    os.makedirs(os.path.join(outpath, part, "foregrounds", patch_name.split("/")[0]), exist_ok=True)

    if "bg_diff" in patch_data:
        bg_diff = patch_data["bg_diff"]
        bg_img = in_img.astype(np.int64) + bg_diff
        Image.fromarray(bg_img.clip(0, 255).astype(np.uint8)).save(
            os.path.join(outpath, part, "backgrounds", patch_name + ".JPEG")
        )

    if "fg_mask" in patch_data:
        x_offs, y_offs = patch_data["fg_off"]
        fg_mask = patch_data["fg_mask"]

        fg_crop = in_img[y_offs : y_offs + fg_mask.shape[0], x_offs : x_offs + fg_mask.shape[1]]
        fg_img = np.concatenate([fg_crop, fg_mask * 255], axis=-1).clip(0, 255).astype(np.uint8)
        Image.fromarray(fg_img).save(os.path.join(outpath, part, "foregrounds", patch_name + ".WEBP"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Patches -> ForNet")
    parser.add_argument("-p", "--patch", type=str, required=True, help="Patch folder")
    parser.add_argument("-in", "--imagenet", type=str, required=True, help="ImageNet folder")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
    parser.add_argument("--keep", action="store_true", help="Keep patch files after extraction")

    args = parser.parse_args()

    max_parallel_workers = multiprocessing.cpu_count()
    if os.environ.get("SLURM_JOB_CPUS_PER_NODE", None):
        max_parallel_workers = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    print(f"INFO: Using {max_parallel_workers} parallel workers")

    for part in ["train", "val"]:
        patch_files = ["val.zip"] if part == "val" else [f"{part}_{i}.zip" for i in range(20)]
        for patch_file_name in tqdm(patch_files, desc=f"processing {part}", position=0, disable=part == "val"):
            if not os.path.exists(os.path.join(args.patch, patch_file_name)):
                tqdm.write(f"INFO: {patch_file_name} not found. Assuming it was already processed...")
                continue
            with zipfile.ZipFile(os.path.join(args.patch, patch_file_name), "r") as patch_file:
                patches = set(patch_file.namelist())
            for p_ in patches:
                if p_.endswith(".pkl") or p_.endswith(".pkl.gz"):
                    ex_name = p_
                    ending = "pkl" if p_.endswith(".pkl") else "pkl.gz"
                    break
            patches = sorted(
                ["/".join(pf.split("/")[-2:]).split(".")[0] for pf in patches if "/" in pf and pf.endswith(ending)]
            )
            patch_name_start = "/".join(ex_name.split("/")[:-2])
            if len(patch_name_start) > 0:
                patch_name_start += "/"

            process_map(
                _extract,
                zip(
                    itertools.repeat(os.path.join(args.patch, patch_file_name)),
                    patches,
                    itertools.repeat(args.output),
                    itertools.repeat(patch_name_start),
                    itertools.repeat(ending),
                    itertools.repeat(args.imagenet),
                    itertools.repeat(part),
                ),
                max_workers=max_parallel_workers,
                desc=f"extracting {patch_file_name}",
                position=1,
                total=len(patches),
            )

            if not args.keep:
                os.remove(os.path.join(args.patch, patch_file_name))
