import os
import json
import pathlib


def ensure(p): os.makedirs(p, exist_ok=True)


def coco_to_yolo_split(split_dir, class_id=0):
    """Creates YOLO .txt files under split_dir/labels for images in split_dir/images using split_dir/coco.json"""
    ann_path = os.path.join(split_dir, "coco.json")
    img_dir = os.path.join(split_dir, "images")
    lab_dir = os.path.join(split_dir, "labels")
    ensure(lab_dir)

    coco = json.load(open(ann_path))
    # map image_id -> (file_name, width, height)
    imginfo = {im["id"]: (im["file_name"], im.get(
        "width"), im.get("height")) for im in coco["images"]}

    # group annotations by image_id
    anns_by_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):  # skip crowd
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    count_files = 0
    for img_id, (fname, w, h) in imginfo.items():
        # allow missing width/height; we can’t normalize without them
        if not (w and h):
            # try to read size from the image file if needed (optional: add PIL)
            # for now, skip if dimensions unknown
            pass

        stem = pathlib.Path(fname).stem
        outp = os.path.join(lab_dir, stem + ".txt")
        lines = []
        for a in anns_by_img.get(img_id, []):
            # Prefer explicit COCO bbox [x, y, width, height] in pixel coords
            if "bbox" in a and a["bbox"]:
                x, y, bw, bh = a["bbox"]
            else:
                # derive bbox from polygon segmentation
                seg = a.get("segmentation")
                if not seg:
                    continue
                # COCO polygon can be list of lists; flatten once
                pts = seg[0] if isinstance(seg[0], (list, tuple)) else seg
                xs = pts[0::2]
                ys = pts[1::2]
                x, y = min(xs), min(ys)
                bw, bh = (max(xs) - x), (max(ys) - y)

            # guard against invalid boxes
            if bw <= 0 or bh <= 0:
                continue

            # normalize to YOLO format
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # clip to [0,1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # write (create empty file if no anns so YOLO knows it's a negative image)
        with open(outp, "w") as f:
            f.write("\n".join(lines))
        count_files += 1

    print(f"[OK] Wrote labels for {count_files} images in: {lab_dir}")


# ---- run for all splits you have ----
base = "/Users/abhin2/Documents/2025 - SEM 1/Engineering_research_project/Model_dataset"
for ds in ["weedset1", "weedset2"]:
    for sp in ["train", "val", "test"]:
        split_dir = os.path.join(base, ds, sp)
        if os.path.exists(os.path.join(split_dir, "coco.json")):
            coco_to_yolo_split(split_dir, class_id=0)
