"""
Tests for document preprocessing pipeline.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing.image_processor import DocumentProcessor
from src.preprocessing.orientation_utils import (
    compute_skew_angle,
    rotate_image,
    detect_text_regions,
    enhance_image
)

@pytest.fixture
def test_image():
    """Create a simple test image."""
    # Create blank image
    img = np.ones((300, 400), dtype=np.uint8) * 255
    
    # Add some text-like features
    cv2.putText(img, "Test Text", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    return img

def test_skew_detection(test_image):
    """Test skew angle detection."""
    # Rotate image by known angle
    angle = 15
    height, width = test_image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(test_image, rotation_matrix, (width, height))
    
    # Detect skew
    detected_angle = compute_skew_angle(rotated)
    
    # Check if detected angle is close to actual angle
    assert abs(detected_angle - angle) < 1.0

def test_text_region_detection(test_image):
    """Test text region detection."""
    regions = detect_text_regions(test_image)
    
    # Should detect at least one region (our test text)
    assert len(regions) > 0
    
    # First region should contain our text
    assert regions[0].shape[0] > 0 and regions[0].shape[1] > 0

def test_image_enhancement(test_image):
    """Test image enhancement."""
    # Add some noise
    noisy = test_image.copy()
    noise = np.random.normal(0, 25, test_image.shape).astype(np.uint8)
    noisy = cv2.add(noisy, noise)
    
    # Enhance image
    enhanced = enhance_image(noisy)
    
    # Enhanced image should have better contrast
    assert cv2.mean(enhanced)[0] != cv2.mean(noisy)[0]

def test_document_processor():
    """Test DocumentProcessor class."""
    processor = DocumentProcessor()
    
    # Test with single image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Test Text", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # Save test image
    test_path = "test_image.png"
    cv2.imwrite(test_path, img)
    
    try:
        # Process document
        processed = processor.process_document(test_path)
        
        # Check results
        assert len(processed) == 1  # One page
        assert len(processed[0]) > 0  # At least one text region
        
    finally:
        # Clean up
        Path(test_path).unlink()