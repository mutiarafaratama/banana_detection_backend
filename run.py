"""
Entry point untuk Flask application
Jalankan file ini untuk start server: python run.py
"""

from app import create_app
import os
from dotenv import load_dotenv
import sys

# Force UTF-8 encoding for standard streams (Windows fix for emojis)
if sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Create Flask app instance
try:
    app = create_app()
    print("✅ Flask app created successfully")
except Exception as e:
    print(f"❌ Error creating Flask app: {str(e)}")
    sys.exit(1)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"🍌 Banana Detection Backend")
    print(f"🚀 Server running on http://{host}:{port}")
    print(f"📝 Environment: {os.getenv('FLASK_ENV', 'production')}")
    print(f"⚙️  Debug mode: {debug}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug
        )
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")
        sys.exit(1)