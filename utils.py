"""
================================================================================
                            UTILS.PY
      Image Processing, Normalization, & Explainability Visualization
================================================================================
Purpose:
    Handles raw image buffer preprocessing into normalized tensors, resizing 
    operations, and generates mock Grad-CAM color-map overlay visualizations 
    to highlight regions indicating fruit rot detection. Integrates OpenCV for 
    image transformation and NumPy for numerical tensor operations.
================================================================================
"""

import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import config


# ============================================================================
# IMAGE PREPROCESSING PIPELINE
# ============================================================================
def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from Streamlit file uploader and convert to RGB numpy array.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        np.ndarray: Image in RGB format (H, W, 3), or None if invalid
    """
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def resize_image(
    image: np.ndarray,
    target_height: int = config.IMAGE_TARGET_HEIGHT,
    target_width: int = config.IMAGE_TARGET_WIDTH,
) -> np.ndarray:
    """
    Resize image to target dimensions using OpenCV INTER_LINEAR interpolation.
    
    Args:
        image: Input image array (H, W, 3)
        target_height: Target height in pixels
        target_width: Target width in pixels
        
    Returns:
        np.ndarray: Resized image (target_height, target_width, 3)
    """
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )
    return resized


def normalize_image_array(
    image: np.ndarray,
    mean: list = config.IMAGE_NORMALIZE_MEAN,
    std: list = config.IMAGE_NORMALIZE_STD,
) -> np.ndarray:
    """
    Apply ImageNet-standard normalization: (x - mean) / std.
    Converts uint8 [0, 255] to float32 [-1, 1] range.
    
    Args:
        image: Input image array (H, W, 3) in uint8 format
        mean: Normalization mean per channel [R, G, B]
        std: Normalization std per channel [R, G, B]
        
    Returns:
        np.ndarray: Normalized float32 array
    """
    image_float = image.astype(np.float32) / 255.0
    mean_array = np.array(mean).reshape(1, 1, 3)
    std_array = np.array(std).reshape(1, 1, 3)
    normalized = (image_float - mean_array) / std_array
    return normalized


def preprocess_image(
    image: np.ndarray,
    target_height: int = config.IMAGE_TARGET_HEIGHT,
    target_width: int = config.IMAGE_TARGET_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline: resize → normalize → add batch dimension.
    
    Args:
        image: Raw image array
        target_height: Target height
        target_width: Target width
        
    Returns:
        Tuple of:
            - Preprocessed tensor (1, H, W, 3) ready for model inference
            - Display-ready image (H, W, 3) for UI rendering
    """
    resized = resize_image(image, target_height, target_width)
    normalized = normalize_image_array(resized)
    batch_tensor = np.expand_dims(normalized, axis=0)
    return batch_tensor, resized


# ============================================================================
# GRAD-CAM & EXPLAINABILITY HEATMAP GENERATION
# ============================================================================
def generate_mock_gradcam_heatmap(
    image: np.ndarray,
    prediction: int,
    confidence: float,
) -> np.ndarray:
    """
    Generate a synthetic Grad-CAM-style heatmap using deterministic image 
    feature extraction (Laplacian + color channel analysis). 
    
    The heatmap highlights regions correlating with:
    - Rotten fruit: Brown/dark spots in RGB channels
    - Fresh fruit: Uniform color distribution
    
    Args:
        image: Original image array (H, W, 3)
        prediction: Predicted class index (0-5)
        confidence: Model confidence score (0-1)
        
    Returns:
        np.ndarray: Heatmap array (H, W, 1) normalized to [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect edges/regions using Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
    
    # Analyze color channel variance for rot detection
    # Rotten fruit has lower saturation and skewed color distribution
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_channel = hsv[:, :, 1] / 255.0
    value_channel = hsv[:, :, 2] / 255.0
    
    # Regions with low saturation + dark areas = likely rot
    rot_likelihood = (1.0 - saturation_channel) * (1.0 - value_channel / 255.0)
    rot_likelihood = cv2.normalize(rot_likelihood, None, 0, 1, cv2.NORM_MINMAX)
    
    # Combine heuristics: edge strength + rot likelihood
    # Weight by confidence to modulate intensity
    heatmap = (0.6 * laplacian_normalized + 0.4 * rot_likelihood) * confidence
    heatmap = np.expand_dims(heatmap, axis=2)
    
    return heatmap


def apply_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay pseudo-colored heatmap on original image using JET colormap.
    Red = high activation (likely rot), Blue = low activation (likely fresh).
    
    Args:
        image: Original image array (H, W, 3) in uint8 [0, 255]
        heatmap: Heatmap array (H, W, 1) in float32 [0, 1]
        alpha: Blending factor (0-1), higher = more heatmap visibility
        colormap: OpenCV colormap constant
        
    Returns:
        np.ndarray: Blended image (H, W, 3) in uint8 format
    """
    # Normalize heatmap to 0-255 range
    heatmap_uint8 = (heatmap.squeeze() * 255).astype(np.uint8)
    
    # Ensure heatmap matches image dimensions
    if heatmap_uint8.shape != image.shape[:2]:
        heatmap_uint8 = cv2.resize(heatmap_uint8, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend images
    blended = cv2.addWeighted(image, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended.astype(np.uint8)


def generate_explainability_visualization(
    original_image: np.ndarray,
    prediction: int,
    confidence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate side-by-side visualization: original image + Grad-CAM overlay.
    
    Args:
        original_image: Original image array (H, W, 3)
        prediction: Predicted class index
        confidence: Model confidence
        
    Returns:
        Tuple of:
            - Original image for left panel
            - Grad-CAM overlaid image for right panel
    """
    heatmap = generate_mock_gradcam_heatmap(original_image, prediction, confidence)
    overlay_image = apply_heatmap_overlay(original_image, heatmap, alpha=0.5)
    
    return original_image, overlay_image


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================
def preprocess_image_batch(
    images: list,
    target_height: int = config.IMAGE_TARGET_HEIGHT,
    target_width: int = config.IMAGE_TARGET_WIDTH,
) -> Tuple[np.ndarray, list]:
    """
    Preprocess multiple images into a batch tensor.
    
    Args:
        images: List of image arrays
        target_height: Target height
        target_width: Target width
        
    Returns:
        Tuple of:
            - Batch tensor (N, H, W, 3)
            - List of resized display images
    """
    batch_list = []
    display_images = []
    
    for img in images:
        tensor, resized = preprocess_image(img, target_height, target_width)
        batch_list.append(tensor[0])  # Remove batch dim, will re-add
        display_images.append(resized)
    
    batch_tensor = np.stack(batch_list, axis=0)
    return batch_tensor, display_images


# ============================================================================
# VALIDATION & ERROR HANDLING
# ============================================================================
def validate_image_array(image: np.ndarray) -> bool:
    """Validate that image array is in expected format."""
    if not isinstance(image, np.ndarray):
        return False
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    if image.dtype not in [np.uint8, np.float32]:
        return False
    return True


def get_image_statistics(image: np.ndarray) -> dict:
    """
    Extract descriptive statistics from image for debugging.
    
    Returns:
        dict: Statistics including mean, std, min, max per channel
    """
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "mean_per_channel": np.mean(image, axis=(0, 1)),
        "std_per_channel": np.std(image, axis=(0, 1)),
        "min_per_channel": np.min(image, axis=(0, 1)),
        "max_per_channel": np.max(image, axis=(0, 1)),
    }
