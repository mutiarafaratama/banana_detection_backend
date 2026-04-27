"""
Flask Application Factory
Initialize Flask app dengan semua konfigurasi
"""

from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.firebase import init_firebase
import os

def create_app(config_class=Config):
    """
    Factory function untuk create Flask app
    """
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Firebase
    init_firebase(app)
    
    # Enable CORS untuk Android app
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "X-Device-Id"]
        }
    })
    
    # Create necessary folders
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
        print(f"✅ Folders created: {app.config['UPLOAD_FOLDER']}, {app.config['MODEL_FOLDER']}")
    except Exception as e:
        print(f"⚠️  Warning creating folders: {str(e)}")
    
    # Register blueprints/routes
    try:
        from app.routes import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
        print("✅ API blueprints registered")
    except Exception as e:
        print(f"⚠️  Warning registering blueprints: {str(e)}")
    
    # Health check endpoint
    @app.route('/')
    def index():
        return {
            'status': 'running',
            'service': 'Banana Detection Backend',
            'version': '1.0.0'
        }
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app