"""
Cloudinary Upload Service
Upload detection results to Cloudinary
"""

import cloudinary
import cloudinary.uploader
from flask import current_app
from datetime import datetime
import io

class CloudinaryService:
    """Service for uploading images to Cloudinary"""
    
    @staticmethod
    def init_cloudinary():
        """Initialize Cloudinary configuration"""
        cloudinary.config(
            cloud_name=current_app.config['CLOUDINARY_CLOUD_NAME'],
            api_key=current_app.config['CLOUDINARY_API_KEY'],
            api_secret=current_app.config['CLOUDINARY_API_SECRET']
        )
    
    @staticmethod
    def upload_image(image_path_or_bytes, folder="detect_banana"):
        """
        Upload image to Cloudinary
        
        Args:
            image_path_or_bytes: File path or bytes
            folder: Cloudinary folder name
            
        Returns:
            dict: Upload result with URL
        """
        try:
            CloudinaryService.init_cloudinary()
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"{folder}/detection_{timestamp}"
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_path_or_bytes,
                public_id=public_id,
                folder=folder,
                resource_type="image"
            )
            
            return {
                'success': True,
                'url': result['secure_url'],
                'public_id': result['public_id'],
                'format': result['format'],
                'width': result['width'],
                'height': result['height']
            }
            
        except Exception as e:
            print(f"❌ Cloudinary upload error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def delete_image(public_id):
        """
        Delete image from Cloudinary
        
        Args:
            public_id: Cloudinary public ID
            
        Returns:
            dict: Delete result
        """
        try:
            CloudinaryService.init_cloudinary()
            
            result = cloudinary.uploader.destroy(public_id)
            
            return {
                'success': result['result'] == 'ok',
                'result': result
            }
            
        except Exception as e:
            print(f"❌ Cloudinary delete error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }