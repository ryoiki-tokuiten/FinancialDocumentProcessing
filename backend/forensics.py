"""
Forensic image and document analysis helpers for the Python execution environment.
Injected directly into the persistent REPL namespace as `forensics`.
"""

import cv2
import numpy as np
import re
from typing import Any, Optional

class ForensicToolkit:
    """Pre-warmed forensic utilities for financial document analysis."""
    
    @staticmethod
    def deskew(image_or_path: Any) -> np.ndarray:
        """
        Detects document skew angle and rotates the image to be straight.
        
        Args:
            image_or_path: numpy ndarray or filepath to image.
            
        Returns:
            Deskewed numpy ndarray image.
        """
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
        else:
            img = image_or_path.copy()
            
        if img is None:
            raise ValueError("Invalid image provided for deskew")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all white pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return img
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
            
        # Rotate image around center
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    @staticmethod
    def ela(image_or_path: Any, quality: int = 90, scale: int = 15) -> np.ndarray:
        """
        Performs Error Level Analysis (ELA) to detect digital tampering, copy-pasting,
        or compression inconsistencies across different regions of an invoice.
        
        Args:
            image_or_path: numpy ndarray or filepath.
            quality: JPEG re-compression quality (default 90).
            scale: Multiplier to amplify pixel difference visibility (default 15).
            
        Returns:
            Enhanced ELA difference image as numpy ndarray.
        """
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
        else:
            img = image_or_path.copy()
            
        if img is None:
            raise ValueError("Invalid image for ELA analysis")
            
        # Re-compress image to memory buffer
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        resaved = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        # Calculate absolute difference
        diff = cv2.absdiff(img, resaved)
        ela_img = cv2.convertScaleAbs(diff, alpha=scale, beta=0)
        return ela_img

    @staticmethod
    def extract_table_grid(image_or_path: Any, kernel_length: int = 30) -> np.ndarray:
        """
        Extracts horizontal and vertical table grid lines using morphological operations.
        Useful for segmenting invoice item rows and column dividers.
        
        Args:
            image_or_path: numpy ndarray or filepath.
            kernel_length: Length of structuring element (default 30).
            
        Returns:
            Binary table grid mask ndarray.
        """
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
        else:
            img = image_or_path.copy()
            
        if img is None:
            raise ValueError("Invalid image for table grid extraction")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Vertical lines kernel
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        vert_lines = cv2.erode(binary, vert_kernel, iterations=1)
        vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=1)
        
        # Horizontal lines kernel
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        horiz_lines = cv2.erode(binary, horiz_kernel, iterations=1)
        horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=1)
        
        # Combine grid
        grid_mask = cv2.addWeighted(vert_lines, 0.5, horiz_lines, 0.5, 0.0)
        _, grid_mask = cv2.threshold(grid_mask, 50, 255, cv2.THRESH_BINARY)
        return grid_mask

    @staticmethod
    def normalize_currency(raw_str: str) -> Optional[float]:
        """
        Parses complex or regional currency expressions into a clean float:
        - Indian Lakhs / Crores ('1.5 Lakhs' -> 150000.0, '2.5 Cr' -> 25000000.0)
        - Indian comma format ('1,05,394.00' -> 105394.0)
        - European decimal comma ('1.250,50 €' -> 1250.50)
        - Historical UK non-decimal currency ('£1. 19s. 6d.' -> 1.975)
        - Standard clean currency ('$1,234.56' -> 1234.56)
        
        Args:
            raw_str: Raw text extracted from document.
            
        Returns:
            Normalized float value or None if unparseable.
        """
        if not raw_str:
            return None
            
        s = str(raw_str).strip()
        
        # 1. Historical UK £ s d (pounds, shillings, pence)
        # e.g., '£1. 19s. 6d.' or '1/19/6' or '1l. 19s. 6d.'
        uk_match = re.search(r'[£l]?\s*(\d+)\s*[.\/-]\s*(\d+)\s*s\b\.?\s*(\d+)?\s*d\b\.?', s, re.IGNORECASE)
        if uk_match:
            pounds = float(uk_match.group(1))
            shillings = float(uk_match.group(2))
            pence = float(uk_match.group(3)) if uk_match.group(3) else 0.0
            return round(pounds + (shillings / 20.0) + (pence / 240.0), 3)
            
        # 2. Indian Lakhs / Crores
        lakh_match = re.search(r'([\d.]+)\s*(?:lakh|lacs|lac)\b', s, re.IGNORECASE)
        if lakh_match:
            return float(lakh_match.group(1)) * 100000.0
            
        cr_match = re.search(r'([\d.]+)\s*(?:crore|cr)\b', s, re.IGNORECASE)
        if cr_match:
            return float(cr_match.group(1)) * 10000000.0
            
        # 3. Clean symbols: remove currency symbols like $, €, £, ₹, Rs., INR
        cleaned = re.sub(r'[^\d.,]', '', s)
        if not cleaned:
            return None
            
        # 4. European format: 1.250,50 -> 1250.50
        if ',' in cleaned and '.' in cleaned:
            if cleaned.rfind(',') > cleaned.rfind('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                cleaned = parts[0] + '.' + parts[1]
            else:
                cleaned = cleaned.replace(',', '')
                
        try:
            return float(cleaned)
        except ValueError:
            return None

# Singleton instance for direct import
forensics = ForensicToolkit()
