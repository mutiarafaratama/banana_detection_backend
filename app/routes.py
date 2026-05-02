"""
API Routes / Endpoints.

Endpoints exposed under /api/*:
  GET  /api/test
  POST /api/predict          (multipart upload)
  POST /api/detect-live      (base64, optimized for camera frames)
  POST /api/feedback         (user correction for re-training queue)
  GET  /api/history          (paginated, filterable)
  DELETE /api/history/<id>
  GET  /api/stats            (aggregated counters)
  GET  /api/stats/daily      (per-day stats)
"""

import base64
import os
from datetime import datetime

from flask import Blueprint, request, jsonify

from app.utils.helpers import (
    allowed_file,
    save_uploaded_file,
    get_device_id,
    parse_float,
    parse_int,
    count_dataset_images,
)
from app.utils.descriptions import (
    enrich_detections,
    summarize_detections,
)
from app.models.yolov8_model import get_detector
from app.services.image_processor import ImageProcessor
from app.services.cloudinary_service import CloudinaryService
from app.services.firebase_service import FirebaseService
from app.services.stats_service import StatsService

api_bp = Blueprint('api', __name__)


# ---------------------------------------------------------------------------
# Health / discovery
# ---------------------------------------------------------------------------

@api_bp.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'model_info': _model_info_payload(),
        'endpoints': {
            'predict':       'POST /api/predict        (multipart: image, save?, device_id?, conf?, iou?, max_det?)',
            'detect_live':   'POST /api/detect-live    (json: image (b64), save?, device_id?, conf?, iou?, max_det?, img_size?)',
            'feedback':      'POST /api/feedback       (json: detection_id, predicted_class, actual_class, ...)',
            'history':       'GET  /api/history        (?limit, cursor, device_id, class)',
            'history_delete':'DELETE /api/history/<id>',
            'stats':         'GET  /api/stats',
            'stats_daily':   'GET  /api/stats/daily?days=7',
            'model_info':    'GET  /api/model-info',
        },
    }), 200


def _model_info_payload():
    """Return a dict with full model metadata and performance stats."""
    return {
        'name':            'Banana Ripeness Detector',
        'architecture':    'YOLOv8n',
        'yolo_version':    'YOLOv8n (Ultralytics)',
        'model_version':   'v2',
        'parameters':      '3M',
        'model_size_mb':   5.9,
        'training': {
            'total_images':      1593,
            'original_images':   670,
            'oversampled_images': 923,
            'epochs':            76,
            'early_stopping':    True,
            'val_images':        167,
            'trained_on':        'Google Colab T4 GPU',
            'training_time':     '~35 menit',
        },
        'classes': ['Mentah', 'Mengkal', 'Matang', 'Busuk'],
        'performance': {
            'mAP50':       0.63,
            'mAP50_label': '63%',
            'per_class': {
                'Mentah':  {'mAP50': 0.83, 'label': '83%'},
                'Mengkal': {'mAP50': 0.66, 'label': '66%'},
                'Matang':  {'mAP50': 0.56, 'label': '56%'},
                'Busuk':   {'mAP50': 0.46, 'label': '46%'},
            },
        },
        'thresholds': {
            'confidence': 0.5,
            'iou':        0.45,
        },
    }


@api_bp.route('/model-info', methods=['GET'])
def model_info():
    """Return model metadata, version, and per-class performance stats."""
    return jsonify({
        'status': 'success',
        **_model_info_payload(),
    }), 200


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _pick(source_dict, *keys):
    """Return the first present value among keys (used to accept both
    snake_case and camelCase field names from different clients)."""
    for k in keys:
        if k in source_dict and source_dict.get(k) is not None:
            return source_dict.get(k)
    return None


def _read_detection_overrides(source_dict):
    """Read optional conf/iou/max_det/img_size/max_box_ratio from a dict.

    Accepts both snake_case (`conf`, `max_det`, `img_size`, `max_box_ratio`)
    and camelCase (`confidence`, `maxDet`, `imgSize`, `maxBoxRatio`) so the
    Android client can use whichever naming convention its data classes
    serialize to without us silently dropping the override.
    """
    return {
        'conf':          parse_float(_pick(source_dict, 'conf', 'confidence'), None),
        'iou':           parse_float(_pick(source_dict, 'iou', 'iouThreshold'), None),
        'max_det':       parse_int(_pick(source_dict, 'max_det', 'maxDet'), 300),
        'img_size':      parse_int(_pick(source_dict, 'img_size', 'imgSize'), 640),
        # Reject any bbox covering more than this fraction of the frame.
        # Helps suppress over-merged "mega cluster" detections that hide
        # smaller per-banana boxes. 1.0 disables the filter.
        'max_box_ratio': parse_float(_pick(source_dict, 'max_box_ratio', 'maxBoxRatio'), None),
    }


def _filter_oversized(detections, image_shape, max_ratio):
    """Drop detections whose bbox area exceeds `max_ratio` of the image area."""
    if not max_ratio or max_ratio >= 1.0 or not image_shape:
        return detections
    try:
        h, w = image_shape[0], image_shape[1]
        frame_area = float(h) * float(w)
        if frame_area <= 0:
            return detections
    except Exception:
        return detections

    kept = []
    for det in detections:
        bbox = det.get('bbox') or {}
        bw = float(bbox.get('width', 0) or (bbox.get('x_max', 0) - bbox.get('x_min', 0)))
        bh = float(bbox.get('height', 0) or (bbox.get('y_max', 0) - bbox.get('y_min', 0)))
        
        # Fallback to normalized coordinates if absolute width/height are missing/zero
        if bw <= 0 or bh <= 0:
            if 'xMin' in bbox and 'xMax' in bbox and 'yMin' in bbox and 'yMax' in bbox:
                bw = float(bbox['xMax'] - bbox['xMin']) * w
                bh = float(bbox['yMax'] - bbox['yMin']) * h
                
        if bw <= 0 or bh <= 0:
            kept.append(det)
            continue
            
        ratio = (bw * bh) / frame_area
        if ratio <= max_ratio:
            kept.append(det)
        else:
            print(f"DEBUG: Dropped oversized box (class={det.get('class')}, ratio={ratio:.2f} > max={max_ratio}). Box w:{bw} h:{bh}, Frame w:{w} h:{h}")
    return kept


def _filter_megabox_low_conf(detections, image_shape,
                              area_threshold=0.50,
                              min_conf_for_large=0.65):
    """Drop low-confidence "mega-box" hallucinations.

    Logic:
      - If a bbox covers <= `area_threshold` of the frame  → keep (small
        per-banana detections always pass).
      - If a bbox covers >  `area_threshold` of the frame  → require
        confidence >= `min_conf_for_large` to keep.

    Rationale: this banana model occasionally outputs a frame-spanning
    bbox at low confidence (e.g. "BUSUK 42% covering 80% of frame"
    when the actual bananas are small in the centre). Such predictions
    are visually overwhelming on the live overlay yet rarely correct.
    Legitimate close-up shots (single big banana with conf >= 65%)
    still pass through unchanged.
    """
    if not image_shape or len(image_shape) < 2:
        return detections
    try:
        h, w = image_shape[0], image_shape[1]
        frame_area = float(h) * float(w)
        if frame_area <= 0:
            return detections
    except Exception:
        return detections

    kept = []
    for det in detections:
        bbox = det.get('bbox') or {}
        # Prefer normalized area_ratio if present (cheaper, no shape math)
        ratio = bbox.get('area_ratio')
        if ratio is None:
            bw = float(bbox.get('width', 0))
            bh = float(bbox.get('height', 0))
            if bw <= 0 or bh <= 0:
                kept.append(det)
                continue
            ratio = (bw * bh) / frame_area

        conf = float(det.get('confidence', 0))
        if ratio > area_threshold and conf < min_conf_for_large:
            print(f"DEBUG: Dropped low-conf mega-box (class={det.get('class')}, "
                  f"area={ratio:.2f}, conf={conf:.2f} < {min_conf_for_large})")
            continue
        kept.append(det)
    return kept


def _build_response(detections, inference_ms, source, image_url, firebase_doc_id, thresholds, image_shape=None):
    summary = summarize_detections(detections)
    image_size = None
    if image_shape and len(image_shape) >= 2:
        image_size = {'width': int(image_shape[1]), 'height': int(image_shape[0])}
        
    print(f"--- API Response [{source}] ---")
    print(f"Image Size (Metadata): {image_size}")
    for i, d in enumerate(detections):
        bbox = d.get('bbox', {})
        print(f"Det {i}: class={d.get('class')}, bbox_norm=(xMin:{bbox.get('xMin')}, yMin:{bbox.get('yMin')}, xMax:{bbox.get('xMax')}, yMax:{bbox.get('yMax')}), bbox_abs=(w:{bbox.get('width')}, h:{bbox.get('height')})")
    print("--------------------------------")

    return {
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat(),
        'source': source,
        'image_url': image_url,
        'image_size': image_size,
        'count': summary['total'],
        'summary': summary,
        'detections': detections,
        'inference_time_ms': inference_ms,
        'inference_time': f"{inference_ms:.1f}ms",
        'thresholds': thresholds,
        'saved': bool(firebase_doc_id),
        'firebase_doc_id': firebase_doc_id,
    }


def _persist(detections, summary, inference_ms, image_url, source, device_id, extras=None, image_shape=None):
    """Save detection event in Firestore. Returns the new doc_id, or None.

    NOTE: We intentionally do NOT bump the legacy aggregated counters
    (`StatsService.record_detection`) anymore. The analytics endpoints
    now compute everything on demand from the saved `detections`
    documents — see `StatsService.get_today_from_detections` /
    `StatsService.get_daily_from_detections`. That guarantees the
    dashboard reflects what's actually persisted, instead of being
    inflated by every live-camera frame the client streams through
    `/api/detect-live` with `save=false`.
    """
    if not image_url:
        # No upload happened (Cloudinary disabled or save=false).
        # Nothing to persist → analytics will simply not include this
        # event, which is the desired behavior.
        return None

    image_size = None
    if image_shape and len(image_shape) >= 2:
        image_size = {'width': int(image_shape[1]), 'height': int(image_shape[0])}

    payload = {
        'device_id': device_id,
        'source': source,
        'image_url': image_url,
        'image_size': image_size,
        'detections': detections,
        'count': summary['total'],
        'dominant_class': summary['dominant_class'],
        'avg_confidence': summary['avg_confidence'],
        'inference_time_ms': inference_ms,
        'model_version': 'v2',
    }
    if extras:
        payload.update(extras)

    save_result = FirebaseService.save_detection(payload)
    return save_result.get('doc_id') if save_result.get('success') else None


# ---------------------------------------------------------------------------
# /api/predict  -- gallery upload
# ---------------------------------------------------------------------------

@api_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type. Allowed: jpg, jpeg, png'}), 400

    device_id = get_device_id()
    overrides = _read_detection_overrides(request.form)
    should_save = request.form.get('save', 'false').lower() == 'true'

    # Turunkan batas confidence khusus untuk upload foto jepretan sendiri
    # (Mengatasi masalah lighting / fokus kamera HP yang berbeda)
    if overrides['conf'] is None:
        overrides['conf'] = 0.25

    filepath = None
    try:
        filepath = save_uploaded_file(file)
        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        image = ImageProcessor.preprocess_for_detection(image_bytes)

        detector = get_detector()
        result = detector.detect(
            image,
            img_size=overrides['img_size'],
            conf=overrides['conf'],
            iou=overrides['iou'],
            max_det=overrides['max_det'],
        )

        # Ubah batas maksimal kotak menjadi 1.0 (100%).
        # Sebelumnya 0.90 (90%) membuat foto "close-up" pisang (diambil dari jarak sangat dekat)
        # otomatis terhapus karena dianggap kotak "raksasa/noise".
        max_box_ratio = overrides['max_box_ratio'] if overrides['max_box_ratio'] is not None else 1.0
        raw_detections = _filter_oversized(result['detections'], result.get('image_shape'), max_box_ratio)
        detections = enrich_detections(raw_detections)
        summary = summarize_detections(detections)
        inference_ms = result['inference_time_ms']

        image_url = None
        firebase_doc_id = None

        if should_save and FirebaseService.is_available():
            cloudinary_result = CloudinaryService.upload_image(filepath)
            if cloudinary_result.get('success'):
                image_url = cloudinary_result['url']
                firebase_doc_id = _persist(
                    detections, summary, inference_ms, image_url,
                    'upload', device_id,
                    extras={'filename': file.filename},
                    image_shape=result.get('image_shape'),
                )
        # When `save=false` (or Firebase isn't configured) we deliberately
        # skip persisting and skip touching analytics counters. The
        # dashboard is now derived from the `detections` collection, so
        # nothing should be counted unless it actually got saved.

        return jsonify(_build_response(
            detections, inference_ms, 'upload', image_url, firebase_doc_id,
            result['thresholds'], image_shape=result.get('image_shape')
        )), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


# ---------------------------------------------------------------------------
# /api/detect-live  -- base64 frames from camera
# ---------------------------------------------------------------------------

@api_bp.route('/detect-live', methods=['POST'])
def detect_live():
    """
    Optimized for live camera frames.

    Default img_size is 416 (vs 640 for upload) for faster inference.
    Mobile can override via "img_size" if it wants higher accuracy.
    Multi-detection works: returns ALL detections above the confidence
    threshold up to max_det (default 300).
    """
    try:
        data = request.get_json(silent=True) or {}
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        base64_image = data['image']
        device_id = get_device_id()
        should_save = bool(data.get('save', False))

        overrides = _read_detection_overrides(data)
        if not data.get('img_size'):
            overrides['img_size'] = 640  # Resolusi penuh agar bisa mendeteksi banyak pisang secara individu
        # Live mode threshold tuning (lihat catatan di bawah).
        if overrides['conf'] is None:
            # 0.30 — cukup tinggi untuk membuang noise, tapi tetap
            # menangkap per-banana box berkonfidensi sedang. Mega-box
            # halusinasi nantinya difilter oleh max_box_ratio (lebih
            # andal daripada hanya mengandalkan conf, karena mega-box
            # kadang punya conf tinggi juga).
            overrides['conf'] = 0.30
        if overrides['iou'] is None:
            # IoU 0.45 + agnostic_nms (default di model) bekerja sama:
            # bbox raksasa lintas-kelas (Mentah hijau + Busuk coklat di
            # area yang sama) akan dilebur jadi satu prediksi tertinggi.
            overrides['iou'] = 0.45

        image = ImageProcessor.preprocess_base64(base64_image)

        detector = get_detector()
        result = detector.detect(
            image,
            img_size=overrides['img_size'],
            conf=overrides['conf'],
            iou=overrides['iou'],
            max_det=overrides['max_det'],
            agnostic_nms=True,
        )

        # Live mode filter pipeline:
        #   1. _filter_megabox_low_conf: drop bbox >50% frame yang
        #      conf-nya <65% (halusinasi mega-box). Bbox close-up
        #      legit (conf tinggi) tetap lolos.
        #   2. _filter_oversized: backstop hard-cap 0.95 untuk bbox
        #      yang benar-benar tidak masuk akal (>95% frame).
        image_shape = result.get('image_shape')
        smart_filtered = _filter_megabox_low_conf(
            result['detections'], image_shape,
            area_threshold=0.50, min_conf_for_large=0.65,
        )
        max_box_ratio = overrides['max_box_ratio'] if overrides['max_box_ratio'] is not None else 0.95
        raw_detections = _filter_oversized(smart_filtered, image_shape, max_box_ratio)
        detections = enrich_detections(raw_detections)
        summary = summarize_detections(detections)
        inference_ms = result['inference_time_ms']

        image_url = None
        firebase_doc_id = None

        if should_save and FirebaseService.is_available():
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            image_bytes = base64.b64decode(base64_image)

            cloudinary_result = CloudinaryService.upload_image(image_bytes)
            if cloudinary_result.get('success'):
                image_url = cloudinary_result['url']
                firebase_doc_id = _persist(
                    detections, summary, inference_ms, image_url,
                    'live_camera', device_id,
                    image_shape=result.get('image_shape'),
                )
        # Live-camera frames with `save=false` are intentionally not
        # counted in analytics — only frames that the user explicitly
        # saves to the database show up in the dashboard.

        return jsonify(_build_response(
            detections, inference_ms, 'live_camera', image_url, firebase_doc_id,
            result['thresholds'], image_shape=result.get('image_shape')
        )), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------------------
# /api/feedback  -- collect user corrections (no auto-retraining)
# ---------------------------------------------------------------------------

@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Body (JSON):
      {
        "detection_id":    "<firestore doc id of the saved detection>",  // optional
        "predicted_class": "Mentah",                                     // required
        "actual_class":    "Matang",                                     // required
        "confidence":      0.82,                                         // optional
        "image_url":       "...",                                        // optional
        "bbox":            { ... },                                      // optional
        "comment":         "kelihatan masih agak hijau"                  // optional
      }
    """
    data = request.get_json(silent=True) or {}
    predicted = data.get('predicted_class')
    actual = data.get('actual_class')

    if not predicted or not actual:
        return jsonify({
            'status': 'error',
            'message': 'predicted_class and actual_class are required',
        }), 400

    payload = {
        'device_id':       get_device_id(),
        'detection_id':    data.get('detection_id'),
        'predicted_class': predicted,
        'actual_class':    actual,
        'is_correct':      predicted == actual,
        'confidence':      parse_float(data.get('confidence'), None),
        'image_url':       data.get('image_url'),
        'bbox':            data.get('bbox'),
        'comment':         data.get('comment'),
        'model_version':   data.get('model_version', 'v1'),
    }

    result = FirebaseService.save_feedback(payload)
    if not result.get('success'):
        return jsonify({
            'status': 'error',
            'message': result.get('error', 'failed_to_save'),
        }), 503 if result.get('error') == 'firebase_not_configured' else 500

    return jsonify({
        'status': 'success',
        'message': 'Terima kasih! Feedback Anda membantu memperbaiki model.',
        'feedback_id': result['doc_id'],
        'is_correct': payload['is_correct'],
    }), 200


# ---------------------------------------------------------------------------
# /api/history
# ---------------------------------------------------------------------------

@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    Query params:
      limit       int (1..100, default 20)
      cursor      ISO timestamp from previous page's `next_cursor`
      device_id   filter to one device (also reads X-Device-Id header)
      class       filter by dominant_class (Mentah / Mengkal / Matang / Busuk)
      mine        "1" to use caller's device_id automatically
    """
    limit = parse_int(request.args.get('limit'), 20)
    cursor = request.args.get('cursor')
    class_filter = request.args.get('class')
    device_id = request.args.get('device_id')

    # Selalu pastikan device_id terisi (dari header X-Device-Id) agar riwayat tidak tercampur
    if not device_id or request.args.get('mine') == '1':
        device_id = get_device_id()

    result = FirebaseService.get_detections(
        limit=limit,
        device_id=device_id,
        class_filter=class_filter,
        cursor=cursor,
    )

    if not result.get('success') and result.get('error') == 'firebase_not_configured':
        return jsonify({
            'status': 'success',
            'detections': [],
            'count': 0,
            'next_cursor': None,
            'note': 'Firebase not configured — history is empty.',
        }), 200

    if not result.get('success'):
        return jsonify({'status': 'error', 'message': result.get('error')}), 500

    # Re-enrich the inner detections so descriptions stay in sync with
    # the latest copy (covers older docs that were saved before enrichment existed)
    for d in result['detections']:
        if isinstance(d.get('detections'), list):
            d['detections'] = enrich_detections(d['detections'])

    return jsonify({
        'status': 'success',
        'detections': result['detections'],
        'count': len(result['detections']),
        'next_cursor': result.get('next_cursor'),
    }), 200


@api_bp.route('/history/<doc_id>', methods=['DELETE'])
def delete_history(doc_id):
    result = FirebaseService.delete_detection(doc_id)
    if not result.get('success'):
        return jsonify({'status': 'error', 'message': result.get('error')}), 500
    return jsonify({'status': 'success', 'deleted_id': doc_id}), 200


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Dashboard statistik HARI INI (WIB) — dihitung langsung dari koleksi
    `detections` di Firestore, bukan dari counter agregat. Ini menjamin
    angka selalu sesuai isi database (tidak membengkak karena live frame
    yang tidak disimpan).

    Field `stats`:
      events_total          → jumlah deteksi yang TERSIMPAN hari ini
                              ("Total Deteksi" di mobile)
      detections_total      → total bbox pisang dari deteksi hari ini
                              ("Pisang Ditemukan" di mobile)
      detections_by_class   → komposisi kelas (Mentah/Matang/Busuk) hari ini
      events_by_source      → upload vs live_camera (yang tersimpan)
      avg_confidence, avg_inference_ms, ... (turunan)
    """
    device_id = get_device_id()

    # Dashboard sekarang selalu fokus pada device pemanggil; tidak ada
    # fallback "global" karena angka all-time bukan yang ingin
    # ditampilkan di mobile.
    stats = StatsService.get_today_from_detections(device_id)

    # Dataset & model performance info (statis, dari training)
    dataset_counts = count_dataset_images()
    total_images = sum(dataset_counts.values())
    model_info = {
        'version': 'v1.0.0 (YOLOv5)',
        'accuracy_metrics': {
            'mAP50': 0.985,      # 98.5% Mean Average Precision
            'precision': 0.962,  # 96.2%
            'recall': 0.946      # 94.6%
        },
        'dataset': {
            'total_images': total_images,
            'split': dataset_counts
        }
    }

    response = {
        'status': 'success',
        'stats': stats,
        'model_info': model_info,
        # Tambahan kecil untuk klien Android: tegaskan periode datanya
        'period': 'today',
        'timezone': 'Asia/Jakarta (UTC+7)',
    }
    if not FirebaseService.is_available():
        response['note'] = 'Firebase not configured — returning empty stats.'
    return jsonify(response), 200


@api_bp.route('/stats/daily', methods=['GET'])
def get_daily_stats():
    """
    Tren deteksi per-hari (WIB), dihitung langsung dari `detections`.
    Default 7 hari (sesuai chart "Tren Deteksi (7 Hari)" di mobile).
    """
    days = parse_int(request.args.get('days'), 7)
    days = max(1, min(days, 60))

    device_id = get_device_id()
    daily = StatsService.get_daily_from_detections(device_id, days=days)
    return jsonify({
        'status': 'success',
        'days': days,
        'timezone': 'Asia/Jakarta (UTC+7)',
        'daily': daily,
    }), 200
