import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os


def process_yolo_mask(image_path, model_path, output_dir):
    """
    Process YOLO segmentation mask to extract and clean up document

    Args:
    image_path (str): Path to input image
    model_path (str): Path to trained YOLO model
    output_dir (str): Directory to save processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)

    # Read the image
    image = cv2.imread(image_path)

    # Run inference
    results = model(image)[0]

    # If no masks detected, return original image
    if results.masks is None or len(results.masks.data) == 0:
        print("No masks detected")
        return None

    # Get the first mask (assuming single document detection)
    mask = results.masks.data[0].cpu().numpy()

    # Convert mask to binary image
    binary_mask = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask to original image size if needed
    if binary_mask.shape[:2] != image.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (main document)
    if not contours:
        print("No contours found")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask and fill the largest contour
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)

    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Create a binary mask
    filled_mask_binary = (filled_mask > 0).astype(np.uint8)

    # Extract document using the mask
    extracted_document = cv2.bitwise_and(image, image, mask=filled_mask_binary)

    # Create a white background
    white_background = np.ones_like(image) * 255

    # Replace black regions with white
    result = np.where(filled_mask_binary[:, :, np.newaxis] == 1, extracted_document, white_background)

    # Generate output paths
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"processed_{filename}")
    mask_output_path = os.path.join(output_dir, f"mask_{filename}")

    # Save processed image and mask
    cv2.imwrite(output_path, result)
    cv2.imwrite(mask_output_path, filled_mask)

    print(f"Processed image saved to {output_path}")
    print(f"Mask saved to {mask_output_path}")

    return result


def process_images_in_directory(input_dir, model_path, output_dir):
    """
    Process all images in a directory using the YOLO model

    Args:
    input_dir (str): Directory containing input images
    model_path (str): Path to trained YOLO model
    output_dir (str): Directory to save processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            image_path = os.path.join(input_dir, filename)
            process_yolo_mask(image_path, model_path, output_dir)


def main():
    # Example usage
    input_image_dir = '../input_images'  # Directory with images to process
    model_path = ('../model/trainedYOLO.pt')  # Path to your trained YOLO model
    output_dir = '../Output1'  # Directory to save processed images

    # Process single image
    # process_yolo_mask('path/to/single/image.jpg', model_path, output_dir)

    # Process all images in a directory
    process_images_in_directory(input_image_dir, model_path, output_dir)


if __name__ == "__main__":
    main()