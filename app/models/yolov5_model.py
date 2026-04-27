"""
YOLOv5 Model Handler - Singleton Pattern
Load model once, reuse for all requests
"""

import torch
import numpy as np
from pathlib import Path
from flask import current_app
import time

class YOLOv5Detector:
    """Singleton class for YOLOv5 model"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super(YOLOv5Detector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model (only once)"""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load YOLOv5 model from weights"""
        try:
            model_path = current_app.config['MODEL_PATH']
            
            print(f"🔄 Loading YOLOv5 model from: {model_path}")
            
            # Check if model exists
            import os
            if not os.path.exists(model_path):
                print(f"⚠️  Model file not found at: {model_path}")
                print(f"ℹ️  For production deployment, download model separately")
                print(f"ℹ️  Model loading will be deferred until first request")
                self._model = None
                return
            
            # Use Ultralytics YOLO instead of torch.hub
            from ultralytics import YOLO
            
            self._model = YOLO(model_path)

            # Move underlying torch model to CPU (Ultralytics YOLO doesn't
            # expose .cpu() directly; thresholds are passed per-call instead)
            try:
                self._model.to('cpu')
            except Exception:
                pass

            print(f"✅ Model loaded successfully!")
            print(f"   - Default confidence threshold: {current_app.config['CONFIDENCE_THRESHOLD']}")
            print(f"   - Default IOU threshold: {current_app.config['IOU_THRESHOLD']}")
            print(f"   - Classes: {current_app.config['CLASS_NAMES']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            # Don't re-raise - allow app to start even if model loading fails
            self._model = None
    
    def detect(self, image, img_size=640, conf=None, iou=None, max_det=300,
               agnostic_nms=True):
        """
        Run detection on a single image.

        Args:
            image:        PIL Image or numpy array
            img_size:     Input image size for model (smaller = faster, less accurate)
            conf:         Optional confidence threshold override (0..1)
            iou:          Optional IoU threshold override (0..1)
            max_det:      Maximum number of objects returned per image
            agnostic_nms: If True, NMS treats all classes as one. Important for
                          this banana model: it tends to predict the same area
                          as both Mentah AND Busuk (different classes) — class-
                          aware NMS keeps both, producing duplicate "mega-box"
                          overlays. Agnostic NMS collapses them to the single
                          highest-confidence prediction.

        Returns:
            dict with detections, inference_time_ms (float), inference_time (str)
        """
        if self._model is None:
            # Try to load the model again if it failed previously
            self._load_model()
            
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Place weights at the configured MODEL_PATH "
                "and restart the server."
            )

        try:
            cfg_conf = current_app.config.get('CONFIDENCE_THRESHOLD', 0.5)
            cfg_iou = current_app.config.get('IOU_THRESHOLD', 0.45)
            conf_val = float(conf) if conf is not None else cfg_conf
            iou_val = float(iou) if iou is not None else cfg_iou

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
                'detections': detections,
                'inference_time_ms': round(inference_ms, 1),
                'inference_time': f"{inference_ms:.1f}ms",
                # Fix: Baca dimensi dari PIL Image (size) jika bukan format numpy array (shape)
                'image_shape': image.shape[:2] if hasattr(image, 'shape') else (image.size[1], image.size[0]) if hasattr(image, 'size') else None,
                'thresholds': {
                    'conf':         conf_val,
                    'iou':          iou_val,
                    'max_det':      int(max_det),
                    'agnostic_nms': bool(agnostic_nms),
                },
            }

        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            raise
    
    def _parse_results(self, results):
        """
        Parse Ultralytics YOLO results to custom format
        
        Args:
            results: Ultralytics YOLO results object
            
        Returns:
            list: List of detections
        """
        detections = []
        class_names = current_app.config['CLASS_NAMES']
        
        # Get results for first image
        result = results[0]
        
        # Original image dimensions for accurate normalization
        orig_h, orig_w = result.orig_shape
        
        # Get boxes, confidences, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max] in original pixels
        confs = result.boxes.conf.cpu().numpy()  # confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # class IDs
        
        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes[i]
            conf = confs[i]
            cls_id = int(class_ids[i])
            
            # Manually calculate normalized coordinates to guarantee accuracy against original shape
            x_min_n = max(0.0, min(1.0, float(x_min / orig_w)))
            y_min_n = max(0.0, min(1.0, float(y_min / orig_h)))
            x_max_n = max(0.0, min(1.0, float(x_max / orig_w)))
            y_max_n = max(0.0, min(1.0, float(y_max / orig_h)))

            # Centre + size in normalized 0..1 space — handy for mobile
            # overlays that draw on a Canvas matching the camera preview.
            w_norm  = max(0.0, x_max_n - x_min_n)
            h_norm  = max(0.0, y_max_n - y_min_n)
            cx_norm = x_min_n + w_norm / 2.0
            cy_norm = y_min_n + h_norm / 2.0

            # Get class name
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min

            detection = {
                'class': class_name,
                'confidence': round(float(conf), 3),
                'bbox': {
                    # Absolute pixel coords in the analysed image
                    'x_min':  int(x_min),
                    'y_min':  int(y_min),
                    'x_max':  int(x_max),
                    'y_max':  int(y_max),
                    'width':  int(width),
                    'height': int(height),

                    # Normalized 0..1 coords (preferred for mobile overlays)
                    'xMin':   round(x_min_n, 6),
                    'yMin':   round(y_min_n, 6),
                    'xMax':   round(x_max_n, 6),
                    'yMax':   round(y_max_n, 6),

                    # Centre + size in normalized space + frame coverage ratio.
                    # Mobile clients can use this to scale boxes elastically
                    # to whatever surface they render on (CameraX preview,
                    # Compose Canvas, etc.).
                    'cx':           round(cx_norm, 6),
                    'cy':           round(cy_norm, 6),
                    'w_norm':       round(w_norm, 6),
                    'h_norm':       round(h_norm, 6),
                    'area_ratio':   round(w_norm * h_norm, 6),
                },
                # Original frame size that these coords are relative to.
                # Letting each detection carry it avoids any ambiguity if the
                # client draws several detections at once.
                'image_size': {
                    'width':  int(orig_w),
                    'height': int(orig_h),
                },
            }
            
            print(f"DEBUG Detection: class={class_name}, conf={conf:.2f}, bbox_norm=({x_min_n:.2f}, {y_min_n:.2f}, {x_max_n:.2f}, {y_max_n:.2f}), bbox_abs=({x_min}, {y_min}, {x_max}, {y_max}), orig_shape=({orig_w}x{orig_h})")

            detections.append(detection)
        
        return detections
    
    @property
    def is_loaded(self):
        """Check if model is loaded"""
        return self._model is not None


# Global instance
detector = None

def get_detector():
    """Get or create detector instance"""
    global detector
    if detector is None:
        detector = YOLOv5Detector()
    return detector