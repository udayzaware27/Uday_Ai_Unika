import os
import shutil
import subprocess
import time

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class FireAlertDetector:
    """Simple fire detector wrapper.

    It loads a YOLO model from a custom path if one is available and reports whether
    a fire/flame class was detected in the frame.
    """

    def __init__(self, model_path=None, conf_threshold=0.70):
        self.model_path = model_path or os.path.join(os.getcwd(), "fire_model.pt")
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        if YOLO is None or not self.model_path or not os.path.exists(self.model_path):
            return
        try:
            self.model = YOLO(self.model_path)
        except Exception:
            self.model = None

    def detect(self, frame, model=None):
        """Return (fire_detected, detections_list)."""
        model_to_use = model or self.model
        if model_to_use is None:
            return False, []

        try:
            results = model_to_use(frame, verbose=False)
        except Exception:
            return False, []

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                    cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                except Exception:
                    continue

                class_name = str(model_to_use.names.get(cls_id, cls_id)).lower()
                if conf < self.conf_threshold:
                    continue

                if "fire" in class_name or "flame" in class_name or "smoke" in class_name:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                    except Exception:
                        continue
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    })

        return bool(detections), detections

    def play_alert_sound(self):
        """Try a system beep or audio tone when fire is detected."""
        try:
            if shutil.which("ffplay"):
                subprocess.Popen(
                    [
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1200:duration=0.5",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                return

            if shutil.which("aplay"):
                subprocess.Popen(
                    ["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                return

            if shutil.which("speaker-test"):
                subprocess.Popen(
                    ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                return

            print("\a")
        except Exception:
            try:
                print("\a")
            except Exception:
                pass

    def draw_alert_overlay(self, frame, detected):
        """Draw a very large and bold middle-screen fire alert."""
        if frame is None or not detected:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        center_x, center_y = w // 2, h // 2

        cv2.rectangle(
            overlay,
            (center_x - 420, center_y - 110),
            (center_x + 420, center_y + 110),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            overlay,
            (center_x - 420, center_y - 110),
            (center_x + 420, center_y + 110),
            (0, 0, 255),
            10,
        )

        label = "DANGER: FIRE DETECTED"
        text_pos = (center_x - 300, center_y + 25)

        cv2.putText(
            overlay,
            label,
            text_pos,
            cv2.FONT_HERSHEY_TRIPLEX,
            1.9,
            (0, 0, 0),
            14,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            text_pos,
            cv2.FONT_HERSHEY_TRIPLEX,
            1.9,
            (0, 0, 255),
            8,
            cv2.LINE_AA,
        )

        alpha = 0.5
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
