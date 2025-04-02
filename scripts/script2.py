import cv2
import numpy as np


def remove_white_background(image):
    """
    Remove white background from the processed image

    Args:
    image (numpy.ndarray): Input image

    Returns:
    numpy.ndarray: Image with white background removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary mask of non-white areas
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def order_points(pts):
    """
    Order points in top-left, top-right, bottom-right, bottom-left order

    Args:
    pts (numpy.ndarray): Array of 4 points

    Returns:
    numpy.ndarray: Ordered points
    """
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point will have the smallest sum
    # Bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right point will have the smallest difference
    # Bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transform to get bird's eye view of document

    Args:
    image (numpy.ndarray): Input image
    pts (numpy.ndarray): Source points

    Returns:
    numpy.ndarray: Transformed image
    """
    # Obtain a consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def process_document(image_path, output_dir):
    """
    Process document: remove white background and dewarp

    Args:
    image_path (str): Path to input image
    output_dir (str): Directory to save processed images
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    image = cv2.imread(image_path)

    # Remove white background
    no_bg_image = remove_white_background(image)

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(no_bg_image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (document)
    if not contours:
        print("No contours found")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)  # Changed from np.int0 to np.int32

    # Dewarp the document
    try:
        dewarped = four_point_transform(no_bg_image, box.reshape(4, 2).astype(np.float32))
    except Exception as e:
        print(f"Error dewarping: {e}")
        return None

    # Generate output paths
    filename = os.path.basename(image_path)
    no_bg_output_path = os.path.join(output_dir, f"no_bg_{filename}")
    dewarped_output_path = os.path.join(output_dir, f"dewarped_{filename}")

    # Save processed images
    cv2.imwrite(no_bg_output_path, no_bg_image)
    cv2.imwrite(dewarped_output_path, dewarped)

    print(f"No background image saved to {no_bg_output_path}")
    print(f"Dewarped image saved to {dewarped_output_path}")

    return dewarped


def process_images_in_directory(input_dir, output_dir):
    """
    Process all images in a directory

    Args:
    input_dir (str): Directory containing input images
    output_dir (str): Directory to save processed images
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            image_path = os.path.join(input_dir, filename)
            process_document(image_path, output_dir)


def main():
    # Example usage
    input_image_dir = '../Next'  # Directory with processed images
    output_dir = '../Final_Output'  # Directory to save final processed images

    # Process single image
    # process_document('path/to/single/image.jpg', output_dir)

    # Process all images in a directory
    process_images_in_directory(input_image_dir, output_dir)


if __name__ == "__main__":
    main()