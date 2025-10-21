"""
Main image processing pipeline for math paper analysis.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path

from .orientation_utils import (
    compute_skew_angle,
    rotate_image,
    detect_text_regions,
    enhance_image
)

class DocumentProcessor:
    """Handles document preprocessing for math paper analysis."""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize document processor.
        
        Args:
            dpi (int): DPI for PDF conversion
        """
        self.dpi = dpi
    
    def load_document(self, file_path: str) -> List[np.ndarray]:
        """
        Load document from file (supports PDF and images).
        
        Args:
            file_path (str): Path to document file
            
        Returns:
            List[np.ndarray]: List of document pages as images
        """
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            # Convert PDF to images
            pages = convert_from_path(file_path, self.dpi)
            return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) 
                   for page in pages]
        else:
            # Load single image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image at {file_path}")
            return [image]
    
    def preprocess_page(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single page.
        
        Args:
            image (np.ndarray): Input page image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect and correct skew
        angle = compute_skew_angle(gray)
        if abs(angle) > 0.5:  # Only rotate if skew is significant
            gray = rotate_image(gray, angle)
        
        # Enhance image quality
        enhanced = enhance_image(gray)
        
        return enhanced
    
    def extract_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract text regions from preprocessed image.
        
        Args:
            image (np.ndarray): Preprocessed grayscale image
            
        Returns:
            List[np.ndarray]: List of text region images
        """
        return detect_text_regions(image)
    
    def process_document(self, file_path: str) -> List[List[np.ndarray]]:
        """
        Process entire document and extract text regions.
        
        Args:
            file_path (str): Path to document file
            
        Returns:
            List[List[np.ndarray]]: List of text regions for each page
        """
        # Load document pages
        pages = self.load_document(file_path)
        
        # Process each page
        processed_pages = []
        for page in pages:
            # Preprocess page
            preprocessed = self.preprocess_page(page)
            
            # Extract text regions
            regions = self.extract_text_regions(preprocessed)
            
            processed_pages.append(regions)
        
        return processed_pages