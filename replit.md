# Banana Detection Backend

Flask REST API untuk deteksi kematangan pisang menggunakan YOLOv8n. Dirancang untuk melayani Android mobile app.

## Architecture

- **Framework**: Flask 2.3.3 (Python 3.11)
- **ML Model**: YOLOv8n (Ultralytics) — `trained_models/banana_detection_v1/weights/best.pt`
- **Classes**: `Mentah (0)`, `Mengkal (1)`, `Matang (2)`, `Busuk (3)`
- **Storage**: Cloudinary (image uploads), Firebase Firestore (history & stats)
- **Server**: Gunicorn (production/Railway), Flask dev server (development/Replit)
- **Deployment**: Railway (via Dockerfile) + GitHub

## Project Structure

```
run.py                          # Entry point (Flask + Gunicorn)
Dockerfile                      # Railway deployment config
Procfile                        # Heroku/Railway fallback
requirements.txt                # Python dependencies (CPU PyTorch)
app/
  __init__.py                   # Flask app factory (v2.0.0)
  config.py                     # Config — CLASS_NAMES=Mentah,Mengkal,Matang,Busuk
  firebase.py                   # Firebase Admin SDK init
  routes.py                     # All API endpoints (model_version=v2)
  models/
    yolov5_model.py             # YOLOv8 singleton detector (class: YOLOv8Detector)
  services/
    cloudinary_service.py       # Image upload to Cloudinary
    firebase_service.py         # Firestore read/write
    history_service.py          # Paginated history
    image_processor.py          # PIL image preprocessing for YOLOv8
    stats_service.py            # Analytics from detections collection
  utils/
    descriptions.py             # 4-class descriptions + enrich_detections()
    helpers.py                  # Utility functions
trained_models/
  banana_detection_v1/
    weights/
      best.pt                   # YOLOv8n weights (v2 — trained on 1593 images)
```

## API Endpoints

All under `/api/`:
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/test` | Health check + endpoint list |
| POST | `/api/predict` | Detect from image upload (multipart) |
| POST | `/api/detect-live` | Detect from base64 camera frame |
| POST | `/api/feedback` | Submit user correction |
| GET | `/api/history` | Paginated detection history |
| DELETE | `/api/history/<id>` | Delete a detection record |
| GET | `/api/stats` | Today's stats (per device) |
| GET | `/api/stats/daily` | Daily stats over N days |

## Detection Response Format

```json
{
  "status": "success",
  "count": 3,
  "detections": [
    {
      "class": "Matang",
      "class_id": 2,
      "confidence": 0.87,
      "confidence_label": "Sangat Yakin",
      "bbox": {
        "xMin": 0.12, "yMin": 0.34, "xMax": 0.56, "yMax": 0.78,
        "cx": 0.34, "cy": 0.56, "w_norm": 0.44, "h_norm": 0.44,
        "area_ratio": 0.19
      },
      "description": { "title": "...", "color_hex": "#FBC02D", ... }
    }
  ],
  "model_version": "v2"
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREBASE_CREDENTIALS_JSON` | Optional | Firebase service account JSON string |
| `FIREBASE_DATABASE_URL` | Optional | Firebase Realtime DB URL |
| `CLOUDINARY_CLOUD_NAME` | Optional | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Optional | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Optional | Cloudinary API secret |
| `MODEL_PATH` | Optional | Override path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | Optional | Default 0.5 |
| `IOU_THRESHOLD` | Optional | Default 0.45 |
| `CLASS_NAMES` | Optional | Default `Mentah,Mengkal,Matang,Busuk` |

## Model Info

- **Architecture**: YOLOv8n (nano, 3M params, 5.9MB)
- **Training**: 1593 images (670 original + 923 oversampled), 76 epochs (early stop)
- **Val set**: 167 original images
- **Results**: mAP50=63%, Mentah=83%, Mengkal=66%, Matang=56%, Busuk=46%
- **Trained on**: Google Colab T4 GPU (~35 menit)

## Development

```bash
python run.py        # port 5000
```

## Notes

- NumPy pinned to `<2.0.0` for PyTorch compatibility
- Firebase & Cloudinary optional — endpoints return graceful fallbacks
- Model loaded as singleton on first request
- `model_version: v2` in all new Firestore documents
