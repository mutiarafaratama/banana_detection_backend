"""
YOLOv8 Model Handler - Singleton Pattern
Load model once, reuse for all requests
Model: YOLOv8n trained on banana ripeness dataset (4 classes)

NOTE: The training dataset had class IDs 0 and 3 accidentally swapped.
The model internally learned: 0=Busuk, 1=Mengkal, 2=Matang, 3=Mentah.
CLASS_NAMES in config is set to match this (Busuk,Mengkal,Matang,Mentah).
The response class_id is remapped to the correct semantic order:
  Mentah=0, Mengkal=1, Matang=2, Busuk=3
"""

import torch
import numpy as np
from pathlib import Path
from flask import current_app
import time


class YOLOv8Detector:
    """Singleton class for YOLOv8 model"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOv8Detector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load YOLOv8 model from weights file"""
        try:
            model_path = current_app.config['MODEL_PATH']

            print(f"Loading YOLOv8 model from: {model_path}")

            import os
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")
                print(f"Place weights at MODEL_PATH and restart the server.")
                self._model = None
                return

            from ultralytics import YOLO
            self._model = YOLO(model_path)

            try:
                self._model.to('cpu')
            except Exception:
                pass

            print(f"YOLOv8 model loaded successfully!")
            print(f"  Confidence threshold : {current_app.config['CONFIDENCE_THRESHOLD']}")
            print(f"  IOU threshold        : {current_app.config['IOU_THRESHOLD']}")
            print(f"  Classes              : {current_app.config['CLASS_NAMES']}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self._model = None

    def detect(self, image, img_size=640, conf=None, iou=None, max_det=300,
               agnostic_nms=True):
        """
        Run detection on a single image.

        Args:
            image:        PIL Image or numpy array
            img_size:     Input image size (640 recommended for YOLOv8n)
            conf:         Confidence threshold override (0..1)
            iou:          IoU threshold override (0..1)
            max_det:      Maximum detections per image
            agnostic_nms: Class-agnostic NMS — collapses overlapping boxes
                          from different classes to the single highest-conf one.
                          Important for tightly-clustered banana bunches.

        Returns:
            dict: detections list, inference_time_ms, thresholds, image_shape
        """
        if self._model is None:
            self._load_model()

        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Place weights at MODEL_PATH and restart."
            )

        try:
            cfg_conf = current_app.config.get('CONFIDENCE_THRESHOLD', 0.5)
            cfg_iou  = current_app.config.get('IOU_THRESHOLD', 0.45)
            conf_val = float(conf) if conf is not None else cfg_conf
            iou_val  = float(iou)  if iou  is not None else cfg_iou

            start_time = time.time()
            results = self._model(
                image,
                imgsz=img_size,
                conf=conf_val,
                iou=iou_val,
                max_det=int(max_det),
                agnostic_nms=bool(agnostic_nms),
                verbose=False,
            )
            inference_ms = (time.time() - start_time) * 1000

            detections = self._parse_results(results)

            return {
                'detections':       detections,
                'inference_time_ms': round(inference_ms, 1),
                'inference_time':   f"{inference_ms:.1f}ms",
                'image_shape':      (
                    image.shape[:2]
                    if hasattr(image, 'shape')
                    else (image.size[1], image.size[0])
                    if hasattr(image, 'size')
                    else None
                ),
                'thresholds': {
                    'conf':         conf_val,
                    'iou':          iou_val,
                    'max_det':      int(max_det),
                    'agnostic_nms': bool(agnostic_nms),
                },
            }

        except Exception as e:
            print(f"Detection error: {str(e)}")
            raise

    # Semantic class IDs exposed in the API response.
    # These are fixed regardless of the model's internal index order.
    SEMANTIC_CLASS_IDS = {
        'Mentah':  0,
        'Mengkal': 1,
        'Matang':  2,
        'Busuk':   3,
    }

    def _parse_results(self, results):
        """
        Parse Ultralytics YOLO results into API-ready format.

        Returns normalized bbox coords (0..1) compatible with both
        Android CameraX overlay and web canvas rendering.

        class_id in the response is the SEMANTIC id (Mentah=0 … Busuk=3),
        not the model's raw internal index (which has IDs 0 and 3 swapped
        due to a dataset preparation error).
        """
        detections  = []
        class_names = current_app.config['CLASS_NAMES']

        result = results[0]
        orig_h, orig_w = result.orig_shape

        boxes     = result.boxes.xyxy.cpu().numpy()
        confs     = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes[i]
            conf   = confs[i]
            cls_id = int(class_ids[i])

            x_min_n = max(0.0, min(1.0, float(x_min / orig_w)))
            y_min_n = max(0.0, min(1.0, float(y_min / orig_h)))
            x_max_n = max(0.0, min(1.0, float(x_max / orig_w)))
            y_max_n = max(0.0, min(1.0, float(y_max / orig_h)))

            w_norm  = max(0.0, x_max_n - x_min_n)
            h_norm  = max(0.0, y_max_n - y_min_n)
            cx_norm = x_min_n + w_norm / 2.0
            cy_norm = y_min_n + h_norm / 2.0

            class_name = (
                class_names[cls_id]
                if cls_id < len(class_names)
                else f"class_{cls_id}"
            )

            # Return the semantic class_id so the mobile app always sees
            # Mentah=0, Mengkal=1, Matang=2, Busuk=3 in the response.
            semantic_id = self.SEMANTIC_CLASS_IDS.get(class_name, cls_id)

            detection = {
                'class':      class_name,
                'class_id':   semantic_id,
                'confidence': round(float(conf), 3),
                'bbox': {
                    'x_min':  int(x_min),
                    'y_min':  int(y_min),
                    'x_max':  int(x_max),
                    'y_max':  int(y_max),
                    'width':  int(x_max - x_min),
                    'height': int(y_max - y_min),
                    'xMin':       round(x_min_n, 6),
                    'yMin':       round(y_min_n, 6),
                    'xMax':       round(x_max_n, 6),
                    'yMax':       round(y_max_n, 6),
                    'cx':         round(cx_norm, 6),
                    'cy':         round(cy_norm, 6),
                    'w_norm':     round(w_norm, 6),
                    'h_norm':     round(h_norm, 6),
                    'area_ratio': round(w_norm * h_norm, 6),
                },
                'image_size': {
                    'width':  int(orig_w),
                    'height': int(orig_h),
                },
            }

            print(
                f"Detection: class={class_name}({cls_id}), conf={conf:.2f}, "
                f"norm=({x_min_n:.2f},{y_min_n:.2f},{x_max_n:.2f},{y_max_n:.2f})"
            )
            detections.append(detection)

        return detections

    @property
    def is_loaded(self):
        return self._model is not None


detector = None


def get_detector():
    """Get or create the singleton YOLOv8 detector."""
    global detector
    if detector is None:
        detector = YOLOv8Detector()
    return detector
