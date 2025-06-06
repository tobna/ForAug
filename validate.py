import argparse
import os
import zipfile
from multiprocessing import Manager, Process

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Validate ForNet dataset.")
parser.add_argument("-f", "--folder", type=str, required=True, help="ForNet folder to work in.")

args = parser.parse_args()


def _file_gather(folder, part, images, classes, q, ret_dict):
    files = []
    ending = "WEBP" if images == "foregrounds" else "JPEG"
    for idx, c in enumerate(classes):
        files.append([f"{c}/{f}" for f in os.listdir(os.path.join(folder, part, images, c)) if f.endswith(ending)])
        q.put((f"{part}/{images}", idx + 1))

    ret_dict[f"{part}/{images}"] = [f for sublist in files for f in sublist]


def _zip_files(folder, part, images, filelist, q):
    with zipfile.ZipFile(os.path.join(folder, f"{images}_{part}.zip"), "w") as zf:
        for idx, file in enumerate(filelist):
            zf.write(os.path.join(folder, part, images, file), file)
            q.put((f"{part}/{images}", idx + 1))


# gather files
classes = [
    f
    for f in os.listdir(os.path.join(args.folder, "train", "foregrounds"))
    if os.path.isdir(os.path.join(args.folder, "train", "foregrounds", f))
]
assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"

print("STEP: Gathering files")

processes = {}
manager = Manager()
files = manager.dict()
update_q = manager.Queue()
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        p = Process(target=_file_gather, args=(args.folder, part, images, classes, update_q, files))
        p.start()
        processes[f"{part}/{images}"] = p

pbars = {}
pos = 0
last_update = {}
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        pbars[f"{part}/{images}"] = tqdm(
            total=len(classes), desc=f"Gathering {part}/{images}", position=pos, smoothing=0.0
        )
        last_update[f"{part}/{images}"] = 0
        pos += 1

running_processes = 4
while running_processes > 0:
    while not update_q.empty():
        folder, idx = update_q.get()
        pbars[folder].update(idx - last_update[folder])
        last_update[folder] = idx
        if idx == len(classes):
            running_processes -= 1
            pbars[folder].refresh()

for pbar in pbars.values():
    pbar.close()

print("STEP: Syncing file lists")
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        folder = f"{part}/{images}"
        processes[folder].join()

print("STEP: Validating number of files")
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        _EXPECTED_FILES = (1_274_557 if images == "foregrounds" else 1_274_556) if part == "train" else 49_751
        assert (
            len(files[f"{part}/{images}"]) == _EXPECTED_FILES
        ), f"{part}/{images}: Expected {_EXPECTED_FILES} files, got {len(files[f'{part}/{images}'])}"

print("DONE: all OK")
