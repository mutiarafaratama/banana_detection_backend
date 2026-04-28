"""
Firebase Firestore Service.

- Initializes the Firebase Admin SDK lazily.
- Becomes a graceful no-op when credentials aren't configured,
  so the API still works (without persistence) on a fresh setup.
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from flask import current_app
from datetime import datetime


class FirebaseService:
    """Service for saving detection results to Firebase."""

    _db = None
    _initialized = False
    _failed = False  # True after a failed init -> stop retrying every request

    # ---- Initialization ----------------------------------------------------

    @classmethod
    def init_firebase(cls):
        """Initialize Firebase Admin SDK if not already initialized."""
        if cls._initialized or cls._failed:
            return

        try:
            cred = cls._build_credentials()
            if cred is None:
                cls._failed = True
                print("ℹ️  Firebase not configured (no credentials). Persistence disabled.")
                return

            options = {}
            db_url = current_app.config.get('FIREBASE_DATABASE_URL')
            if db_url:
                options['databaseURL'] = db_url

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, options)

            cls._db = firestore.client()
            cls._initialized = True
            print("✅ Firebase initialized successfully!")

        except Exception as e:
            cls._failed = True
            print(f"❌ Firebase initialization error: {e}")

    @classmethod
    def _build_credentials(cls):
        """
        Build credentials from one of:
          - FIREBASE_CREDENTIALS_JSON  (raw JSON string in env var)
          - FIREBASE_CREDENTIALS_PATH  (path to service account file)
        Returns a credentials.Certificate or None if nothing is configured.
        """
        cred_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
        if cred_json:
            try:
                info = json.loads(cred_json)
                return credentials.Certificate(info)
            except Exception as e:
                print(f"⚠️  Invalid FIREBASE_CREDENTIALS_JSON: {e}")
                return None

        cred_path = current_app.config.get('FIREBASE_CREDENTIALS_PATH')
        if cred_path and os.path.exists(cred_path):
            return credentials.Certificate(cred_path)

        return None

    @classmethod
    def is_available(cls):
        """Check whether Firestore can be used right now."""
        if not cls._initialized and not cls._failed:
            cls.init_firebase()
        return cls._initialized and cls._db is not None

    @classmethod
    def db(cls):
        """Return raw Firestore client (after ensuring init)."""
        cls.is_available()
        return cls._db

    # ---- Detections --------------------------------------------------------

    @classmethod
    def save_detection(cls, detection_data):
        """Save a detection event document. Returns {success, doc_id|error}."""
        if not cls.is_available():
            return {'success': False, 'error': 'firebase_not_configured'}

        try:
            payload = dict(detection_data)
            payload['timestamp'] = firestore.SERVER_TIMESTAMP
            payload['created_at'] = datetime.utcnow().isoformat()

            doc_ref = cls._db.collection('detections').add(payload)
            return {'success': True, 'doc_id': doc_ref[1].id}

        except Exception as e:
            print(f"❌ Firebase save error: {e}")
            return {'success': False, 'error': str(e)}

    @classmethod
    def get_detections(cls, limit=20, device_id=None, class_filter=None, cursor=None):
        """
        Fetch detections, newest first.

        Args:
            limit:        max docs (1..100)
            device_id:    optional filter on device
            class_filter: optional filter on dominant class
            cursor:       optional ISO timestamp string for pagination
                          (return docs strictly older than this)
        Returns:
            {success, detections, next_cursor}
        """
        if not cls.is_available():
            return {'success': False, 'error': 'firebase_not_configured', 'detections': []}

        try:
            limit = max(1, min(int(limit or 20), 100))
            has_filter = bool(device_id or class_filter)

            query = cls._db.collection('detections')
            if device_id:
                query = query.where('device_id', '==', device_id)
            if class_filter:
                query = query.where('dominant_class', '==', class_filter)

            cursor_dt = None
            if cursor:
                try:
                    cursor_dt = datetime.fromisoformat(cursor.replace('Z', '+00:00'))
                except Exception:
                    cursor_dt = None

            # When filtering with where(), Firestore requires a composite
            # index to also order_by(timestamp DESC). To avoid forcing the
            # user to create the index manually, we fetch unsorted then
            # sort + paginate in Python. Pull a few extra docs so cursor
            # pagination still works.
            if has_filter:
                fetch_n = limit * 5 + 50
                docs = list(query.limit(fetch_n).stream())
            else:
                # No filter -> single-field order_by needs no composite index
                ordered = query.order_by(
                    'timestamp', direction=firestore.Query.DESCENDING
                )
                if cursor_dt:
                    ordered = ordered.start_after({'timestamp': cursor_dt})
                docs = list(ordered.limit(limit).stream())

            items = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                ts = data.get('timestamp')
                if hasattr(ts, 'isoformat'):
                    data['timestamp'] = ts.isoformat()
                items.append(data)

            # In-memory sort + cursor + slice (only path 1 truly needs this)
            if has_filter:
                items.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
                if cursor_dt:
                    cur_iso = cursor_dt.isoformat()
                    items = [x for x in items if (x.get('timestamp') or '') < cur_iso]
                items = items[:limit]

            last_ts_iso = items[-1].get('timestamp') if items else None
            return {
                'success': True,
                'detections': items,
                'next_cursor': last_ts_iso if len(items) == limit else None,
            }

        except Exception as e:
            print(f"❌ Firebase get error: {e}")
            return {'success': False, 'error': str(e), 'detections': []}

    @classmethod
    def get_device_detections_raw(cls, device_id, max_docs=5000):
        """
        Fetch ALL saved detection documents for a device (newest first).

        Used by the analytics endpoints to compute stats directly from the
        actual stored data (instead of from incremental counters that can
        drift / double-count when live-camera streaming is in play).

        Args:
            device_id: device identifier (required for analytics scoping)
            max_docs:  hard cap to keep memory + Firestore reads bounded

        Returns:
            list[dict] — each dict is the saved detection document with
            its `id`, `timestamp` (ISO string), `count`, `detections`,
            `source`, `inference_time_ms`, etc.
        """
        if not cls.is_available() or not device_id:
            return []

        try:
            # Single-field filter on device_id does NOT need a composite
            # index. We sort + filter by date in Python to keep this
            # zero-config for end users.
            query = (
                cls._db.collection('detections')
                .where('device_id', '==', device_id)
                .limit(int(max_docs))
            )
            items = []
            for doc in query.stream():
                data = doc.to_dict() or {}
                data['id'] = doc.id
                ts = data.get('timestamp')
                if hasattr(ts, 'isoformat'):
                    data['timestamp'] = ts.isoformat()
                items.append(data)
            return items
        except Exception as e:
            print(f"❌ Firebase device-detections fetch failed: {e}")
            return []

    @classmethod
    def delete_detection(cls, doc_id):
        if not cls.is_available():
            return {'success': False, 'error': 'firebase_not_configured'}
        try:
            cls._db.collection('detections').document(doc_id).delete()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ---- Feedback ----------------------------------------------------------

    @classmethod
    def save_feedback(cls, feedback_data):
        """Save user feedback document for future model review/retraining."""
        if not cls.is_available():
            return {'success': False, 'error': 'firebase_not_configured'}
        try:
            payload = dict(feedback_data)
            payload['timestamp'] = firestore.SERVER_TIMESTAMP
            payload['created_at'] = datetime.utcnow().isoformat()
            payload.setdefault('status', 'pending_review')

            doc_ref = cls._db.collection('feedbacks').add(payload)
            return {'success': True, 'doc_id': doc_ref[1].id}
        except Exception as e:
            return {'success': False, 'error': str(e)}
