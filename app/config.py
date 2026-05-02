"""
Configuration class untuk Flask app
Semua environment variables dan settings ada di sini
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    
    # Folder Paths
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'trained_models')
    
    # Upload Settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_SIZE', 10 * 1024 * 1024))
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png').split(',')
    
    # Cloudinary Settings
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
    
    # Firebase Settings
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
    FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL')
    
    # Model Settings
    MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(MODEL_FOLDER, 'banana_detection_v1', 'weights', 'best.pt'))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.45))
    
    # Class Names
    # NOTE: Model was trained with IDs 0 and 3 swapped in the dataset.
    # The model visually learned: ID 0 = Busuk, ID 1 = Mengkal, ID 2 = Matang, ID 3 = Mentah.
    # This mapping corrects that so the API response returns the right class names.
    CLASS_NAMES = os.getenv('CLASS_NAMES', 'Busuk,Mengkal,Matang,Mentah').split(',')
    
    @staticmethod
    def init_app(app):
        """Initialize application"""
        pass