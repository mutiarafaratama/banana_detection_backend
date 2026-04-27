"""
Image Preprocessing with OpenCV
Prepare images for YOLOv5 inference
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import base64

class ImageProcessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def preprocess_for_detection(image_bytes, target_size=640):
        """
        Preprocess image for YOLOv5 detection
        
        Args:
            image_bytes: Raw image bytes
            target_size: Target image size
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Honor EXIF orientation tag so a portrait photo from a phone
            # camera is not analysed in landscape (and vice versa).
            # Without this, bbox coords come back relative to the un-rotated
            # sensor frame and the Android overlay draws boxes off-screen.
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            # Convert RGBA / palette to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # No need to resize - YOLOv5 handles this
            return image

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    @staticmethod
    def preprocess_base64(base64_string, target_size=640):
        """
        Preprocess base64 encoded image (for live detection)
        
        Args:
            base64_string: Base64 encoded image
            target_size: Target image size
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Use same preprocessing
            return ImageProcessor.preprocess_for_detection(image_bytes, target_size)
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    @staticmethod
    def enhance_image(image):
        """
        Optional: Enhance image quality
        
        Args:
            image: PIL Image
            
        Returns:
            PIL.Image: Enhanced image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Good for images with varying lighting
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    @staticmethod
    def get_image_info(image):
        """
        Get image information
        
        Args:
            image: PIL Image
            
        Returns:
            dict: Image info (width, height, format, mode)
        """
        return {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode
        }