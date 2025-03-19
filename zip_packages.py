import argparse
import os
import zipfile
from multiprocessing import Manager, Process
from time import sleep

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Zip packages")
parser.add_argument("-f", "--folder", type=str, required=True, help="ForAug folder to work in.")

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
            q.put(("{args.part}/{args.images}", idx + 1))


# maybe all in one file and multiple progress bars: https://stackoverflow.com/questions/77359940/multiple-progress-bars-with-python-multiprocessing

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
        pbars[f"{part}/{images}"] = tqdm(total=len(classes), desc=f"Gathering {part}/{images}", position=pos)
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

for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        _EXPECTED_FILES = (1_274_557 if images == "foregrounds" else 1_274_556) if part == "train" else 49_751
        assert (
            len(files[f"{part}/{images}"]) == _EXPECTED_FILES
        ), f"{part}/{images}: Expected {_EXPECTED_FILES} files, got {len(files[f'{part}/{images}'])}"

print("STEP: Zipping files")
update_q = manager.Queue()
pbars = {}
pos = 0
last_update = {}
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        p = Process(target=_zip_files, args=(args.folder, part, images, list(files[f"{part}/{images}"]), update_q))
        p.start()
        processes[f"{part}/{images}"] = p
        pbars[f"{part}/{images}"] = tqdm(total=len(classes), desc=f"Zipping {part}/{images}", position=pos)
        last_update[f"{part}/{images}"] = 0
        pos += 1

running_processes = 4
while running_processes > 0:
    while not update_q.empty():
        folder, idx = update_q.get()
        pbars[folder].update(idx - last_update[folder])
        last_update[folder] = idx
        if idx == len(files[folder]):
            running_processes -= 1
            pbars[folder].refresh()

for pbar in pbars.values():
    pbar.close()

print("STEP: Finalizing")
for part in ["train", "val"]:
    for images in ["foregrounds", "backgrounds"]:
        folder = f"{part}/{images}"
        processes[folder].join()
