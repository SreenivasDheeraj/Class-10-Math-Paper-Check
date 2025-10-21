import cv2
import numpy as np
from PIL import Image

def detect_orientation(image):
    """
    Detect and correct the orientation of the input document image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Oriented image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # TODO: Implement orientation detection using text line detection
    # or other geometric features
    
    return image

def preprocess_image(image_path):
    """
    Preprocess the input image for feature extraction.
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Detect and correct orientation
    oriented_image = detect_orientation(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(oriented_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # TODO: Add more preprocessing steps as needed
    
    return thresh