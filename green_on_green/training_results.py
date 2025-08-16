import matplotlib.pyplot as plt
import pandas as pd

# point to your results.csv
csv_path = "runs/detect/train4/results.csv"
df = pd.read_csv(csv_path)

# plot training and validation losses
plt.figure()
plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
# plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("YOLO Training & Validation Losses")
plt.show()

# plot precision, recall, mAP50, mAP50-95
plt.figure()
plt.plot(df["epoch"], df["metrics/precision(B)"], label="precision")
plt.plot(df["epoch"], df["metrics/recall(B)"], label="recall")
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.title("YOLO Validation Metrics")
plt.show()
