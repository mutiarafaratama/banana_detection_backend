"""
Helper utilities used across routes/services.
"""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app, request


def allowed_file(filename):
    """Check whether the uploaded filename has an allowed extension."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in current_app.config['ALLOWED_EXTENSIONS']


def save_uploaded_file(file):
    """Save an uploaded file under uploads/ with a unique name."""
    original_filename = secure_filename(file.filename)
    file_ext = original_filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    return filepath


def calculate_confidence_level(confidence):
    """Convert confidence score to coarse label."""
    if confidence >= 0.8:
        return 'High'
    if confidence >= 0.6:
        return 'Medium'
    return 'Low'


def get_device_id():
    """
    Resolve the caller's device id.

    Reads, in order:
      1. X-Device-Id header
      2. device_id form field (multipart)
      3. device_id JSON field
      4. device_id query param
    Falls back to Client IP Address.
    """
    device_id = request.headers.get('X-Device-Id')
    if device_id:
        return device_id.strip()

    if request.form and request.form.get('device_id'):
        return request.form.get('device_id').strip()

    if request.is_json:
        data = request.get_json(silent=True) or {}
        if data.get('device_id'):
            return str(data['device_id']).strip()

    if request.args.get('device_id'):
        return request.args.get('device_id').strip()

    # Fallback menggunakan IP Address perangkat / client
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    else:
        ip = request.remote_addr
        
    return ip or 'anonymous'


def parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def count_dataset_images(base_path='datasets/raw'):
    """Count dataset images in train, val, and test subdirectories."""
    counts = {'training': 0, 'validation': 0, 'testing': 0}
    
    if not os.path.exists(base_path):
        return counts

    valid_exts = {'.jpg', '.jpeg', '.png'}
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                # Use split path to accurately identify folder types instead of just substring
                parts = os.path.normpath(root).lower().split(os.sep)
                if 'train' in parts:
                    counts['training'] += 1
                elif 'val' in parts or 'validation' in parts:
                    counts['validation'] += 1
                elif 'test' in parts or 'testing' in parts:
                    counts['testing'] += 1
                else:
                    # Default to training if not strictly categorized but is in raw
                    counts['training'] += 1
    
    return counts
