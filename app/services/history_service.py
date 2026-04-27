"""
Service layer for handling detection history in Firestore.
Provides functions to save and retrieve detection data scoped by device_id.
"""

from app.firebase import db
from firebase_admin import firestore
from datetime import datetime, timezone

def save_detection(device_id, detection_data):
    """
    Saves a detection result to Firestore under a specific device ID.
    Adds a server timestamp for ordering.
    """
    if not db:
        print("⚠️ Firebase not connected. Skipping save.")
        return None
    if not device_id:
        print("⚠️ Device ID is missing. Skipping save.")
        return None

    try:
        detection_data['timestamp'] = datetime.now(timezone.utc)
        
        # Use a subcollection under the device ID for better data organization
        # Root collection: 'history' -> Document: {device_id} -> Subcollection: 'detections'
        doc_ref = db.collection('history').document(device_id).collection('detections').add(detection_data)
        return doc_ref[1].id # Return the new document ID
    except Exception as e:
        print(f"❌ Error saving detection to Firebase: {str(e)}")
        return None

def get_history_by_device(device_id):
    """
    Retrieves all detection history for a specific device ID, ordered by most recent.
    """
    if not db:
        print("⚠️ Firebase not connected. Cannot get history.")
        return []
    if not device_id:
        print("⚠️ Device ID is missing. Cannot get history.")
        return []

    try:
        detections_ref = db.collection('history').document(device_id).collection('detections').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        history = []
        for doc in detections_ref:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data and hasattr(data['timestamp'], 'isoformat'):
                data['timestamp'] = data['timestamp'].isoformat()
            history.append(data)
        return history
    except Exception as e:
        print(f"❌ Error getting history from Firebase: {str(e)}")
        return []