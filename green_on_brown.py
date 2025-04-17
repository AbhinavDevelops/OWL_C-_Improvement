import cv2
import os
from utils.greenonbrown import GreenOnBrown
import time

# Initialize the GreenOnBrown algorithm
# Change 'exg' to the desired algorithm
algorithm = GreenOnBrown(algorithm='hsv')

# Path to your dataset
dataset_path = 'datasets/my_dataset/'  # Update this to your dataset directory
output_path = 'datasets/output/'  # Directory to save results
os.makedirs(output_path, exist_ok=True)

# Parameters for the algorithm
params = {
    'exg_min': 30,
    'exg_max': 250,
    'hue_min': 30,
    'hue_max': 90,
    'brightness_min': 5,
    'brightness_max': 200,
    'saturation_min': 30,
    'saturation_max': 255,
    'min_detection_area': 100,  # Minimum area for weed detection
    'show_display': True,  # Set to True to visualize results
    'algorithm': 'exg',  # Change to the desired algorithm
    'invert_hue': False,
    'label': 'WEED'
}

inference_time_total = 0
i = 0

# Process each image in the dataset
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)

        i += 1  # increment timer to get average later

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Measure inference time using time.perf_counter()
        start_time = time.perf_counter()  # Start the high-resolution timer

        # Run the GreenOnBrown inference
        contours, boxes, weed_centres, output_image = algorithm.inference(
            image, **params)

        end_time = time.perf_counter()  # End the high-resolution timer
        # Calculate elapsed time and convert to ms
        inference_time = (end_time - start_time)*1000

        inference_time_total += inference_time

        # Save the output image
        output_file = os.path.join(output_path, f"output_{filename}")
        cv2.imwrite(output_file, output_image)

        print(
            f"Processed {filename}: {len(boxes)} weeds detected. Inference time: {inference_time:.6f} ms")

        # Display the result (optional)
        # if params['show_display']:
        #     cv2.imshow("Detection Result", output_image)
        #     cv2.waitKey(0)

print(f"Average time for all inference is: {inference_time_total/i: .6f}")
cv2.destroyAllWindows()
