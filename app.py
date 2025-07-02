from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
import uuid
from pathlib import Path
import shutil
from typing import Optional

app = FastAPI(title="Document Extraction API", version="1.0.0")

# Configuration
MODEL_PATH = "trainedYOLO.pt"
INPUT_DIR = "input"
MASK_DIR = "mask"
EXTRACTED_DIR = "extracted"

# Create necessary directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Load YOLO model at startup
try:
    model = YOLO(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def compress_image_to_target_size(image_path: str, target_size_mb: float = 2.0, min_quality: int = 30):
    """
    Compress image to target size while maintaining reasonable quality

    Args:
        image_path: Path to the image file
        target_size_mb: Target size in MB (default 2MB)
        min_quality: Minimum JPEG quality (default 30)

    Returns:
        str: Path to the compressed image
    """
    target_size_bytes = target_size_mb * 1024 * 1024

    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return image_path

    # Check current file size
    current_size = os.path.getsize(image_path)
    if current_size <= target_size_bytes:
        return image_path  # Already within target size

    # If image has alpha channel (RGBA), convert to RGB for JPEG compression
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Create white background
        rgb_img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

        # Blend with white background using alpha channel
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            rgb_img[:, :, c] = (1 - alpha) * 255 + alpha * img[:, :, c]

        img = rgb_img

    # Start with high quality and reduce until target size is reached
    quality = 95
    temp_path = image_path.replace('.png', '_compressed.jpg')

    while quality >= min_quality:
        # Save with current quality
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Check file size
        if os.path.getsize(temp_path) <= target_size_bytes:
            break

        quality -= 5

    # If still too large, try resizing
    if os.path.getsize(temp_path) > target_size_bytes and quality < min_quality:
        scale_factor = 0.9
        original_img = img.copy()

        while scale_factor > 0.3:  # Don't scale down more than 70%
            # Resize image
            height, width = original_img.shape[:2]
            new_height, new_width = int(height * scale_factor), int(width * scale_factor)
            resized_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Save with minimum quality
            cv2.imwrite(temp_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, min_quality])

            if os.path.getsize(temp_path) <= target_size_bytes:
                break

            scale_factor -= 0.1

    return temp_path


def process_yolo_mask(image_path: str, filename: str) -> tuple[Optional[str], Optional[str]]:
    """
    Process YOLO segmentation mask to extract only the document area (cropped to mask bounds)
    Returns tuple of (extracted_document_path, mask_path) or (None, None) if failed
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    # Run inference
    results = model(image)[0]

    # If no masks detected, return None
    if results.masks is None or len(results.masks.data) == 0:
        return None, None

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
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask and fill the largest contour
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)

    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find bounding box of the mask
    coords = np.column_stack(np.where(filled_mask > 0))
    if len(coords) == 0:
        return None, None

    # Get bounding box coordinates (y, x format from np.where)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add small padding to ensure we don't cut off edges
    padding = 5
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_max = min(image.shape[1], x_max + padding)

    # Crop the original image to the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Crop the mask to the same bounding box
    cropped_mask = filled_mask[y_min:y_max, x_min:x_max]

    # Create RGBA image for transparency
    cropped_rgba = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

    # Set alpha channel based on cropped mask
    cropped_rgba[:, :, 3] = cropped_mask

    # Generate output paths
    name_without_ext = os.path.splitext(filename)[0]

    # Save to specific directories (initial PNG with transparency)
    extracted_path_png = os.path.join(EXTRACTED_DIR, f"{name_without_ext}_extracted.png")
    mask_path = os.path.join(MASK_DIR, f"{name_without_ext}_mask.png")

    # Save initial PNG and mask
    cv2.imwrite(extracted_path_png, cropped_rgba)
    cv2.imwrite(mask_path, filled_mask)

    # Compress the extracted image to ensure it's under 2MB
    compressed_path = compress_image_to_target_size(extracted_path_png, target_size_mb=2.0)

    # Get final file size for logging
    final_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
    print(f"Final extracted image size: {final_size_mb:.2f} MB")

    return compressed_path, mask_path


@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve the main HTML page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Extraction Service</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                font-weight: 600;
            }

            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }

            .content {
                padding: 40px;
            }

            .upload-section {
                margin-bottom: 40px;
            }

            .upload-area {
                border: 3px dashed #e1e5e9;
                padding: 60px 40px;
                text-align: center;
                border-radius: 15px;
                transition: all 0.3s ease;
                cursor: pointer;
                background: #fafbfc;
            }

            .upload-area:hover {
                border-color: #667eea;
                background: #f8f9ff;
                transform: translateY(-2px);
            }

            .upload-area.dragover {
                border-color: #667eea;
                background: #f0f3ff;
            }

            .upload-icon {
                font-size: 3rem;
                color: #667eea;
                margin-bottom: 20px;
            }

            .upload-text {
                font-size: 1.2rem;
                color: #333;
                margin-bottom: 10px;
            }

            .upload-subtext {
                color: #666;
                font-size: 0.9rem;
            }

            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 20px;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 7px 20px rgba(102, 126, 234, 0.4);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 40px;
                background: #f8f9ff;
                border-radius: 15px;
                margin: 20px 0;
            }

            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid #e1e5e9;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .comparison-section {
                display: none;
                margin-top: 40px;
            }

            .comparison-header {
                text-align: center;
                margin-bottom: 30px;
            }

            .comparison-header h2 {
                font-size: 2rem;
                color: #333;
                margin-bottom: 10px;
            }

            .comparison-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }

            .image-card {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }

            .image-card:hover {
                transform: translateY(-5px);
            }

            .image-card h3 {
                font-size: 1.3rem;
                margin-bottom: 15px;
                color: #333;
            }

            .image-card.before h3 {
                color: #dc3545;
            }

            .image-card.after h3 {
                color: #28a745;
            }

            .image-container {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                margin-bottom: 15px;
                background: #fff;
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .comparison-image {
                max-width: 100%;
                max-height: 300px;
                object-fit: contain;
            }

            .download-section {
                text-align: center;
                padding: 30px;
                background: #f8f9ff;
                border-radius: 15px;
            }

            .download-btn {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
            }

            .download-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 7px 20px rgba(40, 167, 69, 0.4);
            }

            @media (max-width: 768px) {
                .comparison-grid {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }

                .header h1 {
                    font-size: 2rem;
                }

                .content {
                    padding: 20px;
                }

                .upload-area {
                    padding: 40px 20px;
                }
            }

            .success-animation {
                animation: slideInUp 0.5s ease-out;
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 Document Extraction Service</h1>
                <p>AI-powered document detection and extraction with precise cropping</p>
            </div>

            <div class="content">
                <div class="upload-section">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="upload-area" id="uploadArea">
                            <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;" required>
                            <div class="upload-icon">📁</div>
                            <div class="upload-text">Click here or drag & drop your document image</div>
                            <div class="upload-subtext">Supported formats: JPG, PNG, BMP, TIFF (Max 10MB)</div>
                        </div>
                        <div style="text-align: center;">
                            <button type="submit" class="btn" id="extractBtn">
                                🚀 Extract Document
                            </button>
                        </div>
                    </form>
                </div>

                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <h3>Processing your document...</h3>
                    <p>Our AI is detecting and extracting your document. This may take a few seconds.</p>
                </div>

                <div id="comparisonSection" class="comparison-section">
                    <div class="comparison-header">
                        <h2>✨ Extraction Results</h2>
                        <p>Before and after comparison of your document</p>
                    </div>

                    <div class="comparison-grid">
                        <div class="image-card before">
                            <h3>🔍 Original Image</h3>
                            <div class="image-container">
                                <img id="originalImage" class="comparison-image" alt="Original image">
                            </div>
                            <p>Your uploaded document image</p>
                        </div>

                        <div class="image-card after">
                            <h3>✅ Extracted Document</h3>
                            <div class="image-container">
                                <img id="extractedImage" class="comparison-image" alt="Extracted document">
                            </div>
                            <p>AI-extracted and cropped document</p>
                        </div>
                    </div>

                    <div class="download-section">
                        <h3>🎉 Perfect! Your document has been extracted</h3>
                        <p style="margin: 10px 0 20px;">Click below to download your cropped document</p>
                        <a id="downloadLink" href="" download class="download-btn">
                            📥 Download Extracted Document
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const uploadForm = document.getElementById('uploadForm');
            const loading = document.getElementById('loading');
            const comparisonSection = document.getElementById('comparisonSection');
            const extractBtn = document.getElementById('extractBtn');

            // Drag and drop functionality
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight(e) {
                uploadArea.classList.add('dragover');
            }

            function unhighlight(e) {
                uploadArea.classList.remove('dragover');
            }

            uploadArea.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                fileInput.files = files;
                updateUploadText(files[0]);
            }

            // Click to upload
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', function(e) {
                if (e.target.files[0]) {
                    updateUploadText(e.target.files[0]);
                }
            });

            function updateUploadText(file) {
                const uploadText = document.querySelector('.upload-text');
                const uploadIcon = document.querySelector('.upload-icon');
                if (file) {
                    uploadText.textContent = `Selected: ${file.name}`;
                    uploadIcon.textContent = '✅';
                } else {
                    uploadText.textContent = 'Click here or drag & drop your document image';
                    uploadIcon.textContent = '📁';
                }
            }

            // Form submission
            uploadForm.addEventListener('submit', async function(e) {
                e.preventDefault();

                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file');
                    return;
                }

                // Check file size (10MB limit)
                if (file.size > 10 * 1024 * 1024) {
                    alert('File size must be less than 10MB');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                // Show loading, hide results
                loading.style.display = 'block';
                comparisonSection.style.display = 'none';
                extractBtn.disabled = true;
                extractBtn.textContent = '⏳ Processing...';

                // Show original image preview
                const originalImageUrl = URL.createObjectURL(file);
                document.getElementById('originalImage').src = originalImageUrl;

                try {
                    const response = await fetch('/extract-document/', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const blob = await response.blob();
                        const extractedImageUrl = URL.createObjectURL(blob);

                        // Show results
                        document.getElementById('extractedImage').src = extractedImageUrl;
                        document.getElementById('downloadLink').href = extractedImageUrl;
                        document.getElementById('downloadLink').download = `extracted_${file.name}`;

                        // Show comparison with animation
                        comparisonSection.style.display = 'block';
                        comparisonSection.classList.add('success-animation');

                        // Smooth scroll to results
                        setTimeout(() => {
                            comparisonSection.scrollIntoView({ behavior: 'smooth' });
                        }, 300);

                    } else {
                        const error = await response.json();
                        alert('Error: ' + error.detail);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                    extractBtn.disabled = false;
                    extractBtn.textContent = '🚀 Extract Document';
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    """Extract document from uploaded image and return only the document area (cropped)"""

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename with timestamp
    timestamp = str(int(time.time()))
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{timestamp}_{file.filename}"

    # Save uploaded file to input directory
    input_path = os.path.join(INPUT_DIR, unique_filename)
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Input image saved to: {input_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        # Extract document using YOLO (saves to extracted/ and mask/ directories)
        extracted_path, mask_path = process_yolo_mask(input_path, unique_filename)
        if not extracted_path:
            raise HTTPException(status_code=400, detail="No document detected in the image")

        print(f"Mask saved to: {mask_path}")
        print(f"Extracted document saved to: {extracted_path}")

        # Return the compressed extracted document
        return FileResponse(
            extracted_path,
            media_type='image/jpeg' if extracted_path.endswith('.jpg') else 'image/png',
            filename=f"extracted_{os.path.splitext(file.filename)[0]}.jpg"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/extract-document-with-mask/")
async def extract_document_with_mask(file: UploadFile = File(...)):
    """Extract document and return both cropped document and mask as JSON response"""

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename with timestamp
    timestamp = str(int(time.time()))
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{timestamp}_{file.filename}"

    # Save uploaded file to input directory
    input_path = os.path.join(INPUT_DIR, unique_filename)
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Input image saved to: {input_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        # Extract document using YOLO (saves to extracted/ and mask/ directories)
        extracted_path, mask_path = process_yolo_mask(input_path, unique_filename)
        if not extracted_path:
            raise HTTPException(status_code=400, detail="No document detected in the image")

        print(f"Mask saved to: {mask_path}")
        print(f"Extracted document saved to: {extracted_path}")

        # Convert images to base64 for JSON response
        import base64

        with open(extracted_path, "rb") as img_file:
            extracted_b64 = base64.b64encode(img_file.read()).decode()

        with open(mask_path, "rb") as mask_file:
            mask_b64 = base64.b64encode(mask_file.read()).decode()

        return {
            "status": "success",
            "extracted_document": f"data:image/jpeg;base64,{extracted_b64}" if extracted_path.endswith(
                '.jpg') else f"data:image/png;base64,{extracted_b64}",
            "mask": f"data:image/png;base64,{mask_b64}",
            "filename": file.filename,
            "compressed": extracted_path.endswith('.jpg'),
            "file_size_mb": round(os.path.getsize(extracted_path) / (1024 * 1024), 2),
            "paths": {
                "input": input_path,
                "mask": mask_path,
                "extracted": extracted_path
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)