from ultralytics.data.converter import convert_coco  # pip install ultralytics
import os
import json
import random
import shutil
import pathlib
random.seed(0)

DATASETS = ["weedset1", "weedset2"]   # <- edit if your folder names differ
COCO_NAME = "weedcoco.json"
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}


def ensure(p): os.makedirs(p, exist_ok=True)
def loadj(p): return json.load(open(p))
def dumpj(x, p): json.dump(x, open(p, "w"))


def unify_to_single_weed(coco):
    coco = dict(coco)
    coco["categories"] = [{"id": 1, "name": "weed"}]
    for a in coco["annotations"]:
        a["category_id"] = 1
    return coco


def split_ids(images):
    imgs = images[:]
    random.shuffle(imgs)
    n = len(imgs)
    n_tr = int(SPLITS["train"]*n)
    n_va = int(SPLITS["val"]*n)
    return {
        "train": {im["id"] for im in imgs[:n_tr]},
        "val":   {im["id"] for im in imgs[n_tr:n_tr+n_va]},
        "test":  {im["id"] for im in imgs[n_tr+n_va:]}
    }


def subjson(coco, idset):
    keep_imgs = [i for i in coco["images"] if i["id"] in idset]
    keep_ids = {i["id"] for i in keep_imgs}
    keep_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    return {"images": keep_imgs, "annotations": keep_anns, "categories": coco["categories"]}


print("Unifying classes → 'weed', splitting, copying images...")
for root in DATASETS:
    coco = unify_to_single_weed(loadj(os.path.join(root, COCO_NAME)))
    ids = split_ids(coco["images"])
    for split in ["train", "val", "test"]:
        out_dir = os.path.join(root, split)
        ensure(os.path.join(out_dir, "images"))
        js = subjson(coco, ids[split])
        dumpj(js, os.path.join(out_dir, "coco.json"))
        img_dir = root
        for im in js["images"]:
            src = os.path.join(img_dir, im["file_name"])
            dst = os.path.join(out_dir, "images",
                               pathlib.Path(im["file_name"]).name)
            ensure(os.path.dirname(dst))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

print("Converting COCO → YOLO...")
for root in DATASETS:
    for split in ["train", "val", "test"]:
        ann = os.path.join(root, split, "coco.json")
        convert_coco(labels_dir=ann, save_dir=os.path.join(
            root, split), use_segments=False)
print("Done.")
