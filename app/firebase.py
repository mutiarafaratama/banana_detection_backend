"""
Firebase Admin SDK initialization.

Credentials are resolved in this order:
  1. FIREBASE_CREDENTIALS_JSON  -- raw JSON content of the service-account key
  2. FIREBASE_CREDENTIALS_PATH  -- path to a service-account .json file
"""

import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

db = None


def _load_credential(app):
    """Return a firebase_admin.credentials.Certificate or None."""
    raw_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if raw_json:
        try:
            cred_dict = json.loads(raw_json)
            return credentials.Certificate(cred_dict), 'env:FIREBASE_CREDENTIALS_JSON'
        except Exception as e:
            print(f"❌ FIREBASE_CREDENTIALS_JSON is set but invalid: {e}")
            return None, None

    cred_path = app.config.get('FIREBASE_CREDENTIALS_PATH')
    if cred_path and os.path.exists(cred_path):
        return credentials.Certificate(cred_path), f'file:{cred_path}'

    return None, None


def init_firebase(app):
    """Initialize Firebase connection."""
    global db
    try:
        cred, source = _load_credential(app)
        if cred is None:
            print("⚠️ Firebase credentials not provided "
                  "(set FIREBASE_CREDENTIALS_JSON or FIREBASE_CREDENTIALS_PATH). "
                  "Firebase not initialized.")
            return

        if not firebase_admin._apps:
            init_options = {}
            db_url = app.config.get('FIREBASE_DATABASE_URL')
            if db_url:
                init_options['databaseURL'] = db_url
            firebase_admin.initialize_app(cred, init_options or None)
            print(f"✅ Firebase App initialized (source: {source}).")

        db = firestore.client()
        print("✅ Firestore client connected.")
    except Exception as e:
        print(f"❌ Error connecting to Firebase: {str(e)}")