"""
Aggregated stats service.

Pattern: instead of scanning every detection document every time
the analytics endpoint is called, we keep small counter documents
that are incremented on each detection.

Documents:
    stats/global               -> totals across all time
    stats_daily/{YYYY-MM-DD}   -> totals per day
"""

from datetime import datetime, timezone, timedelta, date as date_cls
from app.services.firebase_service import FirebaseService

# Local timezone for "hari ini" / per-day bucketing.
# Indonesian app → Western Indonesia Time (WIB, UTC+7).
LOCAL_TZ = timezone(timedelta(hours=7))


class StatsService:
    """Maintain aggregated counters in Firestore."""

    GLOBAL_DOC = ('stats', 'global')
    DAILY_COLLECTION = 'stats_daily'

    @classmethod
    def _firestore(cls):
        """Return Firestore client or None when Firebase is unavailable."""
        if not FirebaseService.is_available():
            return None
        return FirebaseService.db()

    @classmethod
    def record_detection(cls, detections, inference_ms=0.0, source='upload', device_id=None):
        """
        Update aggregated counters for a single detection event.

        Args:
            detections: list of detection dicts (each has 'class', 'confidence')
            inference_ms: float, inference time in milliseconds
            source: 'upload' | 'live_camera'
            device_id: string, optional unique device identifier

        Safe no-op when Firebase isn't configured.
        """
        db = cls._firestore()
        if db is None:
            return

        try:
            # Lazy import so the module works even without firebase-admin
            from firebase_admin import firestore

            # Build per-class delta
            by_class = {}
            confidences = []
            for d in detections:
                cls_name = d.get('class', 'Unknown')
                by_class[cls_name] = by_class.get(cls_name, 0) + 1
                confidences.append(d.get('confidence', 0))

            total_detected = len(detections)
            sum_confidence = sum(confidences)

            # IMPORTANT: For nested fields with set(merge=True), use update() instead.
            # set(merge=True) treats "a.b" as a literal field name with a dot, NOT
            # as a nested path. update() properly interprets dots as nested paths.

            # Per-class confidence sums (for avg confidence per class)
            sum_conf_by_class = {}
            for d in detections:
                cls_name = d.get('class', 'Unknown')
                sum_conf_by_class[cls_name] = (
                    sum_conf_by_class.get(cls_name, 0) + float(d.get('confidence', 0))
                )

            # Did this event detect at least one banana?
            had_detection = 1 if total_detected > 0 else 0

            # ---- GLOBAL COUNTERS ----
            global_ref = db.collection(cls.GLOBAL_DOC[0]).document(cls.GLOBAL_DOC[1])
            global_flat = {
                'events_total': firestore.Increment(1),
                'events_with_detections': firestore.Increment(had_detection),
                'detections_total': firestore.Increment(total_detected),
                'sum_inference_ms': firestore.Increment(float(inference_ms or 0.0)),
                'sum_confidence': firestore.Increment(float(sum_confidence)),
                'last_updated': firestore.SERVER_TIMESTAMP,
            }
            global_nested = {
                f'events_by_source.{source}': firestore.Increment(1),
            }
            for cls_name, count in by_class.items():
                global_nested[f'detections_by_class.{cls_name}'] = firestore.Increment(count)
            for cls_name, sconf in sum_conf_by_class.items():
                global_nested[f'sum_confidence_by_class.{cls_name}'] = firestore.Increment(sconf)

            cls._safe_update(global_ref, global_flat, global_nested)

            # ---- DAILY COUNTERS ----
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            daily_ref = db.collection(cls.DAILY_COLLECTION).document(today)
            daily_flat = {
                'date': today,
                'events_total': firestore.Increment(1),
                'events_with_detections': firestore.Increment(had_detection),
                'detections_total': firestore.Increment(total_detected),
                'sum_inference_ms': firestore.Increment(float(inference_ms or 0.0)),
                'sum_confidence': firestore.Increment(float(sum_confidence)),
                'last_updated': firestore.SERVER_TIMESTAMP,
            }
            daily_nested = {
                f'events_by_source.{source}': firestore.Increment(1),
            }
            for cls_name, count in by_class.items():
                daily_nested[f'detections_by_class.{cls_name}'] = firestore.Increment(count)
            for cls_name, sconf in sum_conf_by_class.items():
                daily_nested[f'sum_confidence_by_class.{cls_name}'] = firestore.Increment(sconf)

            cls._safe_update(daily_ref, daily_flat, daily_nested)

            # ---- DEVICE SPECIFIC COUNTERS ----
            if device_id:
                # Update global stats per device
                dev_ref = db.collection('stats_device').document(device_id)
                cls._safe_update(dev_ref, global_flat, global_nested)
                
                # Update daily stats per device
                dev_daily_ref = dev_ref.collection('daily').document(today)
                cls._safe_update(dev_daily_ref, daily_flat, daily_nested)

        except Exception as e:
            print(f"⚠️  Stats update failed (non-fatal): {e}")

    @staticmethod
    def _safe_update(doc_ref, flat_fields, nested_fields):
        """
        Apply both flat and nested updates to a document.
        - Uses set(merge=True) for the first-time creation of flat fields.
        - Uses update() for nested fields (so dots are treated as paths).
        Falls back to set() if document doesn't exist yet.
        """
        # Ensure doc exists with flat fields (set with merge=True is safe)
        doc_ref.set(flat_fields, merge=True)
        # Now update nested paths properly (dots → nested)
        if nested_fields:
            doc_ref.update(nested_fields)

    @classmethod
    def get_global(cls):
        """Read global stats document. Returns dict or None."""
        db = cls._firestore()
        if db is None:
            return None
        try:
            doc = db.collection(cls.GLOBAL_DOC[0]).document(cls.GLOBAL_DOC[1]).get()
            if not doc.exists:
                return cls._empty_stats()
            data = doc.to_dict()
            return cls._derive(data)
        except Exception as e:
            print(f"⚠️  Stats read failed: {e}")
            return None

    @classmethod
    def get_device_stats(cls, device_id):
        """Read device-specific stats document. Returns dict or None."""
        db = cls._firestore()
        if db is None:
            return None
        try:
            doc = db.collection('stats_device').document(device_id).get()
            if not doc.exists:
                return cls._empty_stats()
            data = doc.to_dict()
            return cls._derive(data)
        except Exception as e:
            print(f"⚠️  Device stats read failed: {e}")
            return None

    @classmethod
    def get_daily(cls, days=7, device_id=None):
        """Return exactly `days` daily stat entries (oldest → newest).

        Hari yang belum ada dokumen di Firestore otomatis di-fill dengan
        zero-stats supaya chart "Tren N Hari" di mobile selalu menampilkan
        semua hari, meski hari ini belum ada deteksi.
        """
        from datetime import timedelta

        # 1) Build daftar tanggal: today UTC mundur (days-1) hari, oldest first
        today = datetime.now(timezone.utc).date()
        date_range = [
            (today - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(days - 1, -1, -1)
        ]

        db = cls._firestore()
        if db is None:
            # Tetap return 7 entry kosong supaya chart konsisten
            return [{**cls._empty_stats(), 'date': d} for d in date_range]

        if device_id:
            collection = db.collection('stats_device').document(device_id).collection('daily')
        else:
            collection = db.collection(cls.DAILY_COLLECTION)

        # 2) Fetch tiap dokumen by ID (efisien — N reads, max 60)
        result = []
        for date_str in date_range:
            try:
                doc = collection.document(date_str).get()
                if doc.exists:
                    data = doc.to_dict() or {}
                    data['date'] = date_str
                    result.append(cls._derive(data))
                else:
                    result.append({**cls._empty_stats(), 'date': date_str})
            except Exception as e:
                print(f"⚠️  Daily stats read failed for {date_str}: {e}")
                result.append({**cls._empty_stats(), 'date': date_str})

        return result

    @staticmethod
    def _unflatten(data):
        """
        Convert legacy flat-dot keys (e.g. "detections_by_class.Busuk": 167)
        into proper nested objects (e.g. "detections_by_class": {"Busuk": 167}).
        Merges with any existing nested values from the same parent.
        """
        result = {}
        for key, value in data.items():
            if '.' in key:
                parent, child = key.split('.', 1)
                # If parent already exists as a nested dict, merge into it.
                if not isinstance(result.get(parent), dict):
                    result[parent] = {}
                # Recurse for deeper nesting (a.b.c)
                if '.' in child:
                    nested = StatsService._unflatten({child: value})
                    result[parent].update(nested)
                else:
                    result[parent][child] = value
            else:
                # Don't overwrite a nested dict already built
                if isinstance(result.get(key), dict) and isinstance(value, dict):
                    result[key].update(value)
                else:
                    result[key] = value
        return result

    @classmethod
    def _derive(cls, data):
        """Compute averages on the fly from accumulated sums.
        Also unflatten any legacy dot-notation keys into nested objects."""
        data = cls._unflatten(data)
        events = data.get('events_total', 0) or 0
        events_with_det = data.get('events_with_detections', 0) or 0
        detections = data.get('detections_total', 0) or 0
        sum_inf = data.get('sum_inference_ms', 0) or 0
        sum_conf = data.get('sum_confidence', 0) or 0

        # Ensure nested fields always exist (mobile expects these keys)
        data.setdefault('detections_by_class', {})
        data.setdefault('events_by_source', {})
        data.setdefault('sum_confidence_by_class', {})

        # Avg confidence per class: sum_confidence_by_class[c] / detections_by_class[c]
        avg_conf_by_class = {}
        per_class_counts = data.get('detections_by_class', {}) or {}
        per_class_sums = data.get('sum_confidence_by_class', {}) or {}
        for cls_name, count in per_class_counts.items():
            sconf = per_class_sums.get(cls_name, 0)
            avg_conf_by_class[cls_name] = round(sconf / count, 3) if count else 0

        return {
            **data,
            'avg_inference_ms': round(sum_inf / events, 1) if events else 0,
            'avg_confidence': round(sum_conf / detections, 3) if detections else 0,
            'avg_confidence_by_class': avg_conf_by_class,
            'avg_detections_per_event': round(detections / events, 2) if events else 0,
            'detection_success_rate': round(events_with_det / events, 3) if events else 0,
            'empty_events': max(0, events - events_with_det),
        }

    @staticmethod
    def _empty_stats():
        return {
            'events_total': 0,
            'events_with_detections': 0,
            'detections_total': 0,
            'detections_by_class': {},
            'events_by_source': {},
            'sum_confidence_by_class': {},
            'avg_inference_ms': 0,
            'avg_confidence': 0,
            'avg_confidence_by_class': {},
            'avg_detections_per_event': 0,
            'detection_success_rate': 0,
            'empty_events': 0,
        }

    # =====================================================================
    # NEW: Analytics computed directly from saved `detections` documents.
    #
    # Why: the legacy counter-based approach above bumps the counters on
    # every `/api/predict` and `/api/detect-live` call — including live
    # camera frames that are NOT saved. That's why the dashboard numbers
    # were "ballooning" (789 / 567 in the screenshots) even though the
    # user had only persisted a handful of detections.
    #
    # The methods below scan the actual stored detection events for the
    # caller's device and compute the same shape on demand. This guarantees
    # the dashboard always reflects what's in the database.
    # =====================================================================

    @classmethod
    def _doc_local_date(cls, doc):
        """
        Return the local (WIB) calendar date of a saved detection doc.

        Prefers the Firestore `timestamp` field (ISO string after fetch);
        falls back to `created_at` (also ISO). Returns None if neither
        parseable.
        """
        for key in ('timestamp', 'created_at'):
            val = doc.get(key)
            if not val:
                continue
            if isinstance(val, str):
                try:
                    dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                except Exception:
                    continue
                if dt.tzinfo is None:
                    # Saved as naive UTC (datetime.utcnow().isoformat()).
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(LOCAL_TZ).date()
            if hasattr(val, 'astimezone'):
                try:
                    return val.astimezone(LOCAL_TZ).date()
                except Exception:
                    continue
        return None

    @staticmethod
    def _aggregate_docs(docs, date_str=None):
        """Compute the analytics shape from a list of saved detection docs."""
        events_total = len(docs)
        events_with_det = 0
        detections_total = 0
        by_class = {}
        sum_conf_by_class = {}
        by_source = {}
        sum_inference_ms = 0.0
        sum_confidence = 0.0  # weighted by per-event count

        for d in docs:
            # `count` is the number of bbox detections in that event
            count = int(d.get('count') or 0)
            detections_total += count
            if count > 0:
                events_with_det += 1

            sum_inference_ms += float(d.get('inference_time_ms') or 0)

            # Aggregate per-class composition from the detections array
            inner = d.get('detections') or []
            if inner:
                for det in inner:
                    cname = det.get('class', 'Unknown')
                    by_class[cname] = by_class.get(cname, 0) + 1
                    conf = float(det.get('confidence') or 0)
                    sum_conf_by_class[cname] = (
                        sum_conf_by_class.get(cname, 0) + conf
                    )
                    sum_confidence += conf

            # Source breakdown (upload vs live_camera)
            src = d.get('source') or 'upload'
            by_source[src] = by_source.get(src, 0) + 1

        avg_conf_by_class = {
            c: round(sum_conf_by_class[c] / n, 3) if n else 0
            for c, n in by_class.items()
        }

        out = {
            'events_total': events_total,
            'events_with_detections': events_with_det,
            'detections_total': detections_total,
            'detections_by_class': by_class,
            'events_by_source': by_source,
            'sum_confidence_by_class': sum_conf_by_class,
            'avg_inference_ms': (
                round(sum_inference_ms / events_total, 1) if events_total else 0
            ),
            'avg_confidence': (
                round(sum_confidence / detections_total, 3) if detections_total else 0
            ),
            'avg_confidence_by_class': avg_conf_by_class,
            'avg_detections_per_event': (
                round(detections_total / events_total, 2) if events_total else 0
            ),
            'detection_success_rate': (
                round(events_with_det / events_total, 3) if events_total else 0
            ),
            'empty_events': max(0, events_total - events_with_det),
        }
        if date_str is not None:
            out['date'] = date_str
        return out

    @classmethod
    def get_today_from_detections(cls, device_id):
        """
        Compute TODAY's stats (WIB) for `device_id` from the saved
        detection documents. Returns the same shape as `_aggregate_docs`.

        When Firebase isn't configured or no device_id provided, returns
        empty stats (so the mobile dashboard still renders).
        """
        today_local = datetime.now(LOCAL_TZ).date()
        empty = {**cls._empty_stats(), 'date': today_local.strftime('%Y-%m-%d')}

        if not device_id or not FirebaseService.is_available():
            return empty

        all_docs = FirebaseService.get_device_detections_raw(device_id)
        today_docs = [
            d for d in all_docs
            if cls._doc_local_date(d) == today_local
        ]
        return cls._aggregate_docs(
            today_docs, date_str=today_local.strftime('%Y-%m-%d')
        )

    @classmethod
    def get_daily_from_detections(cls, device_id, days=7):
        """
        Return `days` per-day stat entries (oldest → newest, WIB) for
        `device_id`, computed from the saved detection documents.

        Days with no saved detection get a zero-filled entry so the
        mobile chart always shows a full N-day range.
        """
        today_local = datetime.now(LOCAL_TZ).date()
        date_range = [
            today_local - timedelta(days=i)
            for i in range(days - 1, -1, -1)
        ]

        if not device_id or not FirebaseService.is_available():
            return [
                {**cls._empty_stats(), 'date': d.strftime('%Y-%m-%d')}
                for d in date_range
            ]

        all_docs = FirebaseService.get_device_detections_raw(device_id)

        # Bucket each doc by its local (WIB) date.
        buckets = {d: [] for d in date_range}
        for doc in all_docs:
            d_local = cls._doc_local_date(doc)
            if d_local in buckets:
                buckets[d_local].append(doc)

        return [
            cls._aggregate_docs(buckets[d], date_str=d.strftime('%Y-%m-%d'))
            for d in date_range
        ]
