# Banana Detection Backend

A Flask REST API for banana ripeness detection using YOLOv5/Ultralytics YOLO. Designed to serve an Android mobile app.

## Architecture

- **Framework**: Flask 2.3.3 (Python)
- **ML Model**: Ultralytics YOLO (YOLOv5 weights at `trained_models/banana_detection_v1/weights/best.pt`)
- **Storage**: Cloudinary (image uploads), Firebase Firestore (detection history & stats)
- **Server**: Gunicorn (production), Flask dev server (development)

## Project Structure

```
run.py                  # Entry point
app/
  __init__.py           # Flask app factory
  config.py             # Configuration (env vars)
  firebase.py           # Firebase Admin SDK init
  routes.py             # All API endpoints
  models/
    yolov5_model.py     # YOLOv5 singleton detector
  services/
    cloudinary_service.py
    firebase_service.py
    history_service.py
    image_processor.py
    stats_service.py
  utils/
    descriptions.py     # Detection enrichment
    helpers.py          # Utility functions
trained_models/
  banana_detection_v1/weights/best.pt  # YOLO weights
```

## API Endpoints

All endpoints are under `/api/`:
- `GET  /api/test`          — Health check
- `POST /api/predict`        — Detect from image upload (multipart)
- `POST /api/detect-live`    — Detect from base64 camera frame
- `POST /api/feedback`       — Submit user correction
- `GET  /api/history`        — Paginated detection history
- `DELETE /api/history/<id>` — Delete a detection record
- `GET  /api/stats`          — Today's stats (per device)
- `GET  /api/stats/daily`    — Daily stats over N days

## Environment Variables

Optional (app works without them, features degraded):
- `FIREBASE_CREDENTIALS_JSON` — Firebase service account JSON string
- `FIREBASE_CREDENTIALS_PATH` — Path to Firebase service account JSON file
- `FIREBASE_DATABASE_URL`     — Firebase Realtime DB URL
- `CLOUDINARY_CLOUD_NAME`     — Cloudinary cloud name
- `CLOUDINARY_API_KEY`        — Cloudinary API key
- `CLOUDINARY_API_SECRET`     — Cloudinary API secret
- `SECRET_KEY`                — Flask secret key
- `MODEL_PATH`                — Override path to YOLO weights
- `CONFIDENCE_THRESHOLD`      — Default confidence (0.5)
- `IOU_THRESHOLD`             — Default IOU (0.45)
- `CLASS_NAMES`               — Comma-separated class names (Mentah,Matang,Busuk)

## Development

Run with: `python run.py` (port 5000)

## Key Notes

- NumPy 2.x compatibility warning from torch is cosmetic — app runs correctly
- Firebase and Cloudinary are optional; endpoints return graceful errors when unconfigured
- ML model loaded lazily on first request if not available at startup
- The app detects banana ripeness into 3 classes: Mentah (unripe), Matang (ripe), Busuk (overripe)
