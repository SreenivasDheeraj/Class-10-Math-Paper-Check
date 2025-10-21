"""
Utilities for document orientation detection and text region segmentation.
"""
import math
import cv2
import numpy as np
from typing import Tuple, List

def compute_skew_angle(image: np.ndarray) -> float:
    """
    Compute the skew angle of the document using text line detection.
    
    Args:
        image (np.ndarray): Grayscale input image
        
    Returns:
        float: Detected skew angle in degrees
    """
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Apply Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return 0.0
    
    # Calculate angles and find the dominant one
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) % 180
        if 0 <= angle <= 45 or 135 <= angle <= 180:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Get the median angle
    median_angle = np.median(angles)
    if median_angle > 45:
        median_angle = median_angle - 90
        
    return median_angle

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by the given angle.
    
    Args:
        image (np.ndarray): Input image
        angle (float): Rotation angle in degrees
        
    Returns:
        np.ndarray: Rotated image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

def detect_text_regions(image: np.ndarray) -> List[np.ndarray]:
    """
    Detect and extract text regions from the document.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        List[np.ndarray]: List of extracted text region images
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort text regions
    regions = []
    min_area = gray.shape[0] * gray.shape[1] * 0.001  # Min 0.1% of image area
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > min_area:
            region = gray[y:y+h, x:x+w]
            regions.append(region)
    
    # Sort regions top to bottom
    regions.sort(key=lambda x: cv2.boundingRect(
        cv2.findNonZero(cv2.threshold(x, 0, 255, cv2.THRESH_BINARY)[1])
    )[1])
    
    return regions

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better text recognition.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        np.ndarray: Enhanced image
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoising(image)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced