"""
Preprocessing module for math paper analysis.
"""
from .image_processor import DocumentProcessor
from .orientation_utils import (
    compute_skew_angle,
    rotate_image,
    detect_text_regions,
    enhance_image
)