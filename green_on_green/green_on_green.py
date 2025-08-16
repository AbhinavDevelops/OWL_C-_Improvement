import cv2
import numpy as np
import tensorflow as tf

# ───── CONFIGURATION ─────
VIDEO_PATH = '../datasets/stuart_data/GOPR1088.mp4'
MODEL_PATH = 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
INPUT_SIZE = (224, 224)  # Adjust to match your model
LABELS_PATH = 'labels.txt'  # Optional for classification

# ───── LOAD MODEL ─────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ───── LOAD LABELS ─────
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    labels = None

# ───── OPEN VIDEO ─────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video file")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to model input size
    input_frame = cv2.resize(frame, INPUT_SIZE)
    input_tensor = np.expand_dims(input_frame, axis=0).astype(np.float32)

    # Optional: normalize if required
    if input_details[0]['dtype'] == np.float32:
        input_tensor = input_tensor / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process output (for classification)
    if output_data.ndim == 2:
        class_id = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        label = labels[class_id] if labels and class_id < len(
            labels) else f"Class {class_id}"
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('TFLite Inference (CPU)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
