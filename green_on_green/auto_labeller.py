# predict_and_save.py
# Draw YOLO predictions onto images and save to ./predicted

import os
import pathlib
from ultralytics import YOLO
import cv2  # pip install opencv-python

MODEL = "runs/detect/train4/weights/best.pt"  # <- change if needed
SOURCE = "/Users/abhin2/Documents/2025 - SEM 1/Engineering_research_project/Model_dataset/weedset1/val/images"
OUTDIR = "./predicted"  # final folder where annotated images are saved

CONF = 0.30     # tweak if you want fewer/more boxes
IOU = 0.60
IMGSZ = 640
DEVICE = "mps"  # "cpu" if no Apple GPU


def iter_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    p = pathlib.Path(root)
    if p.is_file() and p.suffix.lower() in exts:
        yield str(p)
        return
    for f in p.rglob("*"):
        if f.suffix.lower() in exts:
            yield str(f)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    model = YOLO(MODEL)

    # stream=True yields one Results per image; r.plot() returns an annotated ndarray (BGR)
    for img_path in iter_images(SOURCE):
        results = model.predict(
            source=img_path,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=300,
            device=DEVICE,
            save=False,
            stream=True,
            verbose=False,
            batch=1
        )
        for r in results:
            vis = r.plot()  # annotated image (numpy array, BGR)
            stem = pathlib.Path(r.path).stem
            out_path = os.path.join(OUTDIR, f"{stem}_pred.jpg")
            cv2.imwrite(out_path, vis)
            print("saved:", out_path)


if __name__ == "__main__":
    main()
