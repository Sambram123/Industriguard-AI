from ultralytics import YOLO
import cv2

class PPEDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        print(f"[PPEDetector] Model loaded → {model_path}")

        # Update these class names once you have a trained PPE model
        self.HELMET_CLASSES  = ["helmet", "hard hat", "hardhat"]
        self.VEST_CLASSES    = ["vest", "safety vest", "reflective vest"]
        self.PERSON_CLASSES  = ["person"]

    def detect(self, frame):
        """Runs detection and returns list of detected objects"""
        results     = self.model(frame, verbose=False)
        detections  = []

        for result in results:
            for box in result.boxes:
                class_id    = int(box.cls[0])
                confidence  = float(box.conf[0])
                class_name  = self.model.names[class_id]
                bbox        = box.xyxy[0].tolist()

                detections.append({
                    "class_id":   class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "bbox":       [int(b) for b in bbox]
                })

        return detections

    def detect_with_tracks(self, frame, tracker="bytetrack.yaml"):
        """
        Runs detection + tracking using Ultralytics trackers (ByteTrack by default).
        Returns detections like detect(), but includes:
          - track_id (int) when available
        """
        try:
            results = self.model.track(frame, persist=True, tracker=tracker, verbose=False)
        except Exception:
            # Fallback to plain detect if track() is not supported in this env
            return self.detect(frame)

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                track_id = None
                # Ultralytics typically exposes tracking IDs via box.id
                try:
                    if getattr(box, "id", None) is not None:
                        track_id = int(box.id[0])
                except Exception:
                    track_id = None

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": [int(b) for b in bbox],
                    "track_id": track_id,
                })

        return detections

    def detect_with_tracks_fast(self, frame, tracker="bytetrack.yaml", imgsz=None):
        """
        Same as detect_with_tracks(), but allows reducing inference size via imgsz.
        """
        try:
            kwargs = {"persist": True, "tracker": tracker, "verbose": False}
            if imgsz:
                kwargs["imgsz"] = int(imgsz)
            results = self.model.track(frame, **kwargs)
        except Exception:
            return self.detect(frame)

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                track_id = None
                try:
                    if getattr(box, "id", None) is not None:
                        track_id = int(box.id[0])
                except Exception:
                    track_id = None

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": [int(b) for b in bbox],
                    "track_id": track_id,
                })

        return detections

    def _is_class(self, det, class_names):
        name = (det.get("class_name") or "").lower()
        return any(c in name for c in class_names)

    def split_detections(self, detections):
        """Splits detections into persons / helmets / vests / other."""
        persons, helmets, vests, other = [], [], [], []
        for d in detections or []:
            if self._is_class(d, self.PERSON_CLASSES):
                persons.append(d)
            elif self._is_class(d, self.HELMET_CLASSES):
                helmets.append(d)
            elif self._is_class(d, self.VEST_CLASSES):
                vests.append(d)
            else:
                other.append(d)
        return persons, helmets, vests, other

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _point_in_bbox(self, pt, bbox):
        x, y = pt
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def per_person_compliance(self, detections):
        """
        Computes PPE compliance per detected person by associating helmet/vest boxes
        whose center falls inside the person bbox.
        Returns a list:
          [{ person_det, has_helmet, has_vest, safety_percentage }, ...]
        """
        persons, helmets, vests, _ = self.split_detections(detections)
        out = []

        for p in persons:
            pb = p["bbox"]
            has_helmet = any(self._point_in_bbox(self._center(h["bbox"]), pb) for h in helmets)
            has_vest   = any(self._point_in_bbox(self._center(v["bbox"]), pb) for v in vests)
            safety_pct = int(round(((int(has_helmet) + int(has_vest)) / 2) * 100))
            out.append({
                "person_det": p,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "safety_percentage": safety_pct
            })

        return out

    def check_ppe_compliance(self, detections):
        """
        Checks helmet and vest presence.
        Returns simple compliance dict.
        """
        detected_names = " ".join(
            [d["class_name"].lower() for d in detections]
        )

        has_helmet = any(
            word in detected_names
            for word in self.HELMET_CLASSES
        )
        has_vest = any(
            word in detected_names
            for word in self.VEST_CLASSES
        )

        missing = []
        if not has_helmet:
            missing.append("Helmet")
        if not has_vest:
            missing.append("Safety Vest")

        return {
            "has_helmet": has_helmet,
            "has_vest":   has_vest,
            "missing":    missing
        }

    def draw_boxes(self, frame, detections):
        """Draws detection boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class_name"]} {det["confidence"]}'

            is_violation = any(
                word in det["class_name"].lower()
                for word in ["no_helmet", "no_vest", "no helmet", "no vest"]
            )
            color = (0, 0, 255) if is_violation else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        return frame