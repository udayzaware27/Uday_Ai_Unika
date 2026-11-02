#!/usr/bin/env python3
"""
AI Video Analytics - Final
Two independent use cases in one Tkinter application:

1) PARKING ANALYTICS
   - Select a video
   - Add editable 4-corner Parking Slot polygons
   - Drag individual corners
   - Drag the whole polygon
   - Change edge width/color
   - Save/load parking-slot JSON
   - Select a YOLO vehicle model
   - Run YOLO on the selected video
   - Calculate parking-slot occupancy
   - FREE = green, OCCUPIED = red
   - Shows P1/P2/... and FREE/OCCUPIED counters

2) YOLO MODEL TEST
   - Select any trained/retrained .pt model
   - Automatically reads model classes
   - Select a video
   - Run the model on the selected video
   - Save named tests and compare two models
   - Review test history, metrics, and annotated output videos
   - Bounding boxes + class + confidence
   - Class filter
   - Confidence / IoU controls

Requirements:
    pip install -r requirements.txt

IMPORTANT:
Do NOT name this file tkinter.py. That shadows Python's tkinter module.
"""

import json
import math
import os
import queue
import shutil
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, colorchooser
    from tkinter import ttk
except Exception as exc:
    raise RuntimeError(
        "Tkinter could not be imported. On macOS use a Python distribution "
        "that includes Tk. Also make sure this script is NOT named tkinter.py."
    ) from exc

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None


APP_TITLE = "AI Video Analytics - Parking + YOLO Test"
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_OVERLAP = 0.30
DEFAULT_EDGE_WIDTH = 3
MAX_DISPLAY_W = 1050
MAX_DISPLAY_H = 650
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def video_information(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not read video information:\n{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frames": frame_count,
        "duration": round(frame_count / fps, 2) if fps > 0 else 0,
    }


def run_saved_model_test(model_path, video_path, output_folder, confidence, iou):
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        )

    model = YOLO(str(model_path))
    started = time.perf_counter()
    predictions = model.predict(
        source=str(video_path),
        conf=confidence,
        iou=iou,
        save=True,
        project=str(output_folder),
        name="prediction",
        exist_ok=True,
        verbose=False,
        device=choose_device(),
        stream=True,
    )

    names = model_names(model)
    class_counts = Counter()
    total_detections = 0
    for result in predictions:
        if result.boxes is None:
            continue
        for class_id in result.boxes.cls.detach().cpu().numpy().astype(int):
            name = names.get(class_id, str(class_id))
            class_counts[name] += 1
            total_detections += 1

    processing_time = time.perf_counter() - started
    prediction_folder = Path(output_folder) / "prediction"
    output_video = None
    if prediction_folder.is_dir():
        output_video = next(
            (
                path
                for path in prediction_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ),
            None,
        )

    return {
        "name": Path(model_path).name,
        "total_detections": total_detections,
        "class_counts": dict(class_counts),
        "processing_time": round(processing_time, 2),
        "output_video": (
            str(output_video.relative_to(Path(output_folder)))
            if output_video is not None
            else None
        ),
    }


def choose_device():
    """Use Apple MPS when available, otherwise CPU."""
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def model_names(model):
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    return abs(float(cv2.contourArea(pts)))


def point_in_polygon(point, points):
    contour = np.asarray(points, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def polygon_intersection_ratio(slot_points, box_xyxy):
    """
    Intersection area / parking-slot area.

    This answers:
    "What percentage of the parking slot is covered by a detected vehicle?"
    """
    slot = np.asarray(slot_points, dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    rect = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )

    slot_area = polygon_area(slot)
    if slot_area <= 0:
        return 0.0

    try:
        inter_area, _ = cv2.intersectConvexConvex(slot, rect)
        return max(0.0, min(1.0, float(inter_area) / slot_area))
    except Exception:
        # Fallback: mask intersection.
        min_x = int(max(0, math.floor(min(slot[:, 0].min(), x1))))
        min_y = int(max(0, math.floor(min(slot[:, 1].min(), y1))))
        max_x = int(math.ceil(max(slot[:, 0].max(), x2)))
        max_y = int(math.ceil(max(slot[:, 1].max(), y2)))
        if max_x <= min_x or max_y <= min_y:
            return 0.0

        w = max_x - min_x + 1
        h = max_y - min_y + 1
        a = np.zeros((h, w), np.uint8)
        b = np.zeros((h, w), np.uint8)

        s = np.round(slot - [min_x, min_y]).astype(np.int32)
        r = np.round(rect - [min_x, min_y]).astype(np.int32)

        cv2.fillPoly(a, [s], 1)
        cv2.fillPoly(b, [r], 1)
        inter = np.logical_and(a, b).sum()
        return float(inter) / max(1, np.count_nonzero(a))


def hex_to_bgr(value):
    value = value.lstrip("#")
    if len(value) != 6:
        return (0, 255, 0)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _right_click_bindings(widget):
    if widget.tk.call("tk", "windowingsystem") == "aqua":
        return ("<Button-2>", "<Button-3>", "<Control-Button-1>")
    return ("<Button-3>",)


class VideoSource:
    """Small wrapper around OpenCV video opening."""

    def __init__(self):
        self.cap = None
        self.path = None
        self.fps = 25.0
        self.frame_count = 0
        self.width = 0
        self.height = 0

    def open(self, path):
        self.release()
        self.path = path

        cap = None
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
            except Exception:
                cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video:\n{path}")

        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if self.fps <= 0:
            self.fps = 25.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read_at(self, index):
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class VideoCanvas(ttk.Frame):
    """Reusable video canvas with frame rendering and timeline."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.canvas = tk.Canvas(self, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.photo = None
        self.last_frame = None
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.source_width = 0
        self.source_height = 0

    def show_frame(self, frame, keep_bgr=True):
        if frame is None:
            return

        if keep_bgr:
            self.last_frame = frame.copy()

        h, w = frame.shape[:2]
        self.source_width = w
        self.source_height = h

        max_w = max(100, self.winfo_width() - 4)
        max_h = max(100, self.winfo_height() - 4)

        scale = min(max_w / w, max_h / h)
        scale = min(1.0, scale)

        dw = max(1, int(w * scale))
        dh = max(1, int(h * scale))

        resized = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image)

        self.canvas.delete("video")
        self.offset_x = (max_w - dw) // 2 + 2
        self.offset_y = (max_h - dh) // 2 + 2
        self.display_scale = scale

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.photo,
            anchor="nw",
            tags="video",
        )

    def source_to_canvas(self, x, y):
        return (
            self.offset_x + x * self.display_scale,
            self.offset_y + y * self.display_scale,
        )

    def canvas_to_source(self, x, y):
        if self.display_scale <= 0:
            return 0.0, 0.0
        return (
            (x - self.offset_x) / self.display_scale,
            (y - self.offset_y) / self.display_scale,
        )


class VideoController:
    """
    Shared video playback engine.

    The UI thread owns Tk widgets.
    A worker thread performs optional YOLO inference.
    """

    def __init__(self, owner, video_widget, timeline, status_var):
        self.owner = owner
        self.video_widget = video_widget
        self.timeline = timeline
        self.status_var = status_var

        self.source = VideoSource()
        self.frame_index = 0
        self.playing = False
        self.worker = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)

        self.speed = 1.0
        self.infer_function = None
        self.on_frame = None
        self.mode_name = ""

        self.timeline.bind("<ButtonRelease-1>", self._timeline_release)

    def open(self, path):
        self.stop()
        self.source.open(path)
        self.frame_index = 0
        self.timeline.configure(
            from_=0,
            to=max(0, self.source.frame_count - 1),
        )
        frame = self.source.read_at(0)
        if frame is None:
            raise RuntimeError("The video opened but the first frame could not be read.")
        self.display_frame(frame)
        self._update_status()

    def display_frame(self, frame):
        if self.on_frame:
            frame = self.on_frame(frame, self.frame_index)
        self.video_widget.show_frame(frame)
        self.timeline.set(self.frame_index)

    def _update_status(self):
        total = max(0, self.source.frame_count - 1)
        self.status_var.set(
            f"Frame {self.frame_index + 1}/{total + 1}   |   "
            f"{self.source.width}×{self.source.height}   |   "
            f"{self.source.fps:.1f} FPS"
        )

    def play(self, infer_function=None, mode_name=""):
        if self.source.cap is None:
            return

        self.infer_function = infer_function
        self.mode_name = mode_name

        if self.playing:
            return

        self.playing = True
        self.stop_event.clear()

        if infer_function is not None:
            self.worker = threading.Thread(
                target=self._inference_worker,
                daemon=True,
            )
            self.worker.start()
            self.owner.after(30, self._poll_queue)
        else:
            self.worker = threading.Thread(
                target=self._plain_play_worker,
                daemon=True,
            )
            self.worker.start()
            self.owner.after(15, self._poll_queue)

    def pause(self):
        self.playing = False
        self.stop_event.set()

    def stop(self):
        self.playing = False
        self.stop_event.set()
        self.infer_function = None
        self.worker = None

        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.frame_index = 0
        if self.source.cap is not None:
            frame = self.source.read_at(0)
            if frame is not None:
                self.display_frame(frame)
                self._update_status()

    def next_frame(self):
        self.pause()
        if self.source.frame_count:
            self.frame_index = min(
                self.source.frame_count - 1,
                self.frame_index + 1,
            )
            frame = self.source.read_at(self.frame_index)
            if frame is not None:
                self.display_frame(frame)
                self._update_status()

    def previous_frame(self):
        self.pause()
        if self.source.frame_count:
            self.frame_index = max(0, self.frame_index - 1)
            frame = self.source.read_at(self.frame_index)
            if frame is not None:
                self.display_frame(frame)
                self._update_status()

    def seek(self, index):
        if self.source.frame_count <= 0:
            return
        self.pause()
        self.frame_index = max(
            0,
            min(self.source.frame_count - 1, int(float(index))),
        )
        frame = self.source.read_at(self.frame_index)
        if frame is not None:
            self.display_frame(frame)
            self._update_status()

    def set_speed(self, speed):
        self.speed = float(speed)

    def _plain_play_worker(self):
        frame_delay = 1.0 / max(1.0, self.source.fps)

        while self.playing and not self.stop_event.is_set():
            next_index = self.frame_index + 1
            if next_index >= self.source.frame_count:
                self.playing = False
                break

            frame = self.source.read_at(next_index)
            if frame is None:
                self.playing = False
                break

            try:
                self.frame_queue.put_nowait((next_index, frame))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait((next_index, frame))
                except queue.Full:
                    pass

            time.sleep(frame_delay / max(0.1, self.speed))

    def _inference_worker(self):
        # The selected model is expected to be loaded before this worker starts.
        infer = self.infer_function
        while self.playing and not self.stop_event.is_set():
            next_index = self.frame_index + 1
            if next_index >= self.source.frame_count:
                self.playing = False
                break

            frame = self.source.read_at(next_index)
            if frame is None:
                self.playing = False
                break

            try:
                result = infer(frame, next_index)
                self.frame_queue.put_nowait((next_index, result))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait((next_index, result))
                except Exception:
                    pass
            except Exception as exc:
                self.owner.after(
                    0,
                    lambda e=str(exc): self.owner.handle_worker_error(e),
                )
                self.playing = False
                break

    def _poll_queue(self):
        got = False

        while True:
            try:
                index, item = self.frame_queue.get_nowait()
            except queue.Empty:
                break

            got = True
            self.frame_index = index

            if self.infer_function is not None:
                frame = item
            else:
                frame = item

            self.display_frame(frame)
            self._update_status()

        if self.playing:
            self.owner.after(20, self._poll_queue)
        elif not got:
            self._update_status()

    def _timeline_release(self, _event):
        value = self.timeline.get()
        self.seek(int(float(value)))


class ParkingEditor:
    """
    Four-corner editable polygon editor.

    Every parking slot is ALWAYS classed as "Parking Slot".
    """

    HANDLE_RADIUS = 7
    HIT_DISTANCE = 14

    def __init__(self, app, video_widget):
        self.app = app
        # The parking video canvas belongs to ParkingTab. Do not look for
        # app.parking_video because that attribute does not exist.
        self.video = video_widget

        self.slots = []
        self.selected = None
        self.drag_mode = None
        self.drag_handle = None
        self.drag_last = None

        self.edge_width_var = tk.IntVar(value=DEFAULT_EDGE_WIDTH)
        self.color_var = tk.StringVar(value="#00ff00")

        self.video.canvas.bind("<Button-1>", self._mouse_down)
        self.video.canvas.bind("<B1-Motion>", self._mouse_move)
        self.video.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.context_menu = tk.Menu(self.video.canvas, tearoff=0)
        self.context_menu.add_command(
            label="Duplicate",
            command=self.app.parking_editor_duplicate,
        )
        self.context_menu.add_command(
            label="Delete",
            command=self.app.parking_editor_delete,
        )
        for sequence in _right_click_bindings(self.video.canvas):
            self.video.canvas.bind(sequence, self._show_context_menu)

    def add_slot(self):
        if self.video.last_frame is None:
            messagebox.showwarning(
                "No video",
                "Select a video first, then add a parking slot.",
            )
            return

        w = self.video.source_width
        h = self.video.source_height

        cx = w * 0.5
        cy = h * 0.5
        bw = max(80, w * 0.12)
        bh = max(45, h * 0.07)

        points = [
            [cx - bw / 2, cy - bh / 2],
            [cx + bw / 2, cy - bh / 2],
            [cx + bw / 2, cy + bh / 2],
            [cx - bw / 2, cy + bh / 2],
        ]

        slot_id = f"P{len(self.slots) + 1}"
        self.slots.append(
            {
                "id": slot_id,
                "class": "Parking Slot",
                "points": points,
                "color": self.color_var.get(),
                "edge_width": int(self.edge_width_var.get()),
            }
        )
        self.selected = len(self.slots) - 1
        self.redraw()

    def delete_selected(self):
        if self.selected is None:
            return
        if not self.slots:
            return

        del self.slots[self.selected]

        for i, slot in enumerate(self.slots, start=1):
            slot["id"] = f"P{i}"

        self.selected = None if not self.slots else min(
            self.selected,
            len(self.slots) - 1,
        )
        self.redraw()

    def clear(self):
        self.slots.clear()
        self.selected = None
        self.redraw()

    def set_edge_width(self):
        if self.selected is None:
            return
        self.slots[self.selected]["edge_width"] = max(
            1,
            int(self.edge_width_var.get()),
        )
        self.redraw()

    def choose_color(self):
        color = colorchooser.askcolor(
            title="Choose Parking Slot Color",
            initialcolor=self.color_var.get(),
        )
        if color and color[1]:
            self.color_var.set(color[1])
            if self.selected is not None:
                self.slots[self.selected]["color"] = color[1]
            self.redraw()

    def select_slot(self, index):
        if index is None:
            return
        if 0 <= index < len(self.slots):
            self.selected = index
            slot = self.slots[index]
            self.color_var.set(slot.get("color", "#00ff00"))
            self.edge_width_var.set(int(slot.get("edge_width", 3)))
            self.redraw()

    def duplicate_selected(self):
        if self.selected is None or not 0 <= self.selected < len(self.slots):
            return

        source = self.slots[self.selected]
        duplicate = source.copy()
        duplicate["id"] = f"P{len(self.slots) + 1}"
        duplicate["points"] = [list(point) for point in source["points"]]
        self.slots.append(duplicate)
        self.selected = len(self.slots) - 1
        self.color_var.set(duplicate.get("color", "#00ff00"))
        self.edge_width_var.set(int(duplicate.get("edge_width", 3)))
        self.redraw()

    def _hit_test(self, x, y):
        if self.video.display_scale <= 0:
            return None, None

        # Work in source coordinates.
        sx, sy = self.video.canvas_to_source(x, y)

        # First check handles of the selected slot.
        if self.selected is not None and self.selected < len(self.slots):
            points = self.slots[self.selected]["points"]
            for i, (px, py) in enumerate(points):
                d = math.hypot(sx - px, sy - py)
                if d <= self.HIT_DISTANCE / self.video.display_scale:
                    return self.selected, i

        # Then check polygons from topmost to bottommost.
        for idx in range(len(self.slots) - 1, -1, -1):
            if point_in_polygon((sx, sy), self.slots[idx]["points"]):
                return idx, -1

        return None, None

    def _show_context_menu(self, event):
        index, _ = self._hit_test(event.x, event.y)
        if index is None:
            return "break"

        self.select_slot(index)
        self.context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _mouse_down(self, event):
        index, handle = self._hit_test(event.x, event.y)

        if index is None:
            self.selected = None
            self.redraw()
            return

        self.selected = index
        slot = self.slots[index]

        if handle >= 0:
            self.drag_mode = "corner"
            self.drag_handle = handle
        else:
            self.drag_mode = "move"

        self.drag_last = self.video.canvas_to_source(event.x, event.y)

        self.color_var.set(slot.get("color", "#00ff00"))
        self.edge_width_var.set(int(slot.get("edge_width", 3)))
        self.redraw()

    def _mouse_move(self, event):
        if self.selected is None or self.drag_mode is None:
            return

        current = self.video.canvas_to_source(event.x, event.y)
        last = self.drag_last
        dx = current[0] - last[0]
        dy = current[1] - last[1]

        slot = self.slots[self.selected]
        w = self.video.source_width
        h = self.video.source_height

        if self.drag_mode == "corner":
            p = slot["points"][self.drag_handle]
            p[0] = max(0, min(w - 1, current[0]))
            p[1] = max(0, min(h - 1, current[1]))
        elif self.drag_mode == "move":
            new_points = []
            for px, py in slot["points"]:
                new_points.append(
                    [
                        max(0, min(w - 1, px + dx)),
                        max(0, min(h - 1, py + dy)),
                    ]
                )
            slot["points"] = new_points

        self.drag_last = current
        self.redraw()

    def _mouse_up(self, _event):
        self.drag_mode = None
        self.drag_handle = None
        self.drag_last = None

    def redraw(self):
        if self.video.last_frame is None:
            return

        frame = self.video.last_frame.copy()
        self.draw_overlays(frame)
        self.video.show_frame(frame, keep_bgr=False)

    def draw_overlays(self, frame):
        for idx, slot in enumerate(self.slots):
            points = np.asarray(slot["points"], dtype=np.float32)
            pts = np.round(points).astype(np.int32).reshape((-1, 1, 2))

            color = hex_to_bgr(slot.get("color", "#00ff00"))
            width = int(slot.get("edge_width", 3))

            cv2.polylines(
                frame,
                [pts],
                True,
                color,
                width,
                cv2.LINE_AA,
            )

            # Selection highlight.
            if idx == self.selected:
                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    (255, 255, 255),
                    max(1, width + 2),
                    cv2.LINE_AA,
                )
                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    color,
                    width,
                    cv2.LINE_AA,
                )

            # Label near first point.
            x, y = int(points[0][0]), int(points[0][1])
            cv2.putText(
                frame,
                slot["id"] + " - Parking Slot",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            if idx == self.selected:
                for px, py in points:
                    cv2.circle(
                        frame,
                        (int(px), int(py)),
                        self.HANDLE_RADIUS,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        (int(px), int(py)),
                        self.HANDLE_RADIUS,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

    def save_json(self):
        if self.video.source_width <= 0 or self.video.source_height <= 0:
            messagebox.showwarning("No source", "Select a parking video first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Parking Slots",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        data = {
            "version": 1,
            "source_width": self.video.source_width,
            "source_height": self.video.source_height,
            "rois": self.slots,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.app.parking_json_path.set(path)
        messagebox.showinfo(
            "Saved",
            f"Parking-slot configuration saved:\n{path}",
        )

    def load_json(self):
        path = filedialog.askopenfilename(
            title="Load Parking Slots",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_w = float(data.get("source_width", 0))
            source_h = float(data.get("source_height", 0))

            current_w = self.video.source_width
            current_h = self.video.source_height

            sx = current_w / source_w if source_w and current_w else 1.0
            sy = current_h / source_h if source_h and current_h else 1.0

            loaded = []
            for i, item in enumerate(data.get("rois", []), start=1):
                points = item.get("points", [])
                if len(points) != 4:
                    continue

                scaled = [
                    [float(p[0]) * sx, float(p[1]) * sy]
                    for p in points
                ]

                loaded.append(
                    {
                        "id": f"P{i}",
                        "class": "Parking Slot",
                        "points": scaled,
                        "color": item.get("color", "#00ff00"),
                        "edge_width": int(item.get("edge_width", 3)),
                    }
                )

            self.slots = loaded
            self.selected = 0 if loaded else None
            self.app.parking_json_path.set(path)

            if loaded:
                self.color_var.set(loaded[0]["color"])
                self.edge_width_var.set(loaded[0]["edge_width"])

            self.redraw()

        except Exception as exc:
            messagebox.showerror(
                "Load error",
                f"Could not load parking JSON:\n{exc}",
            )


class ParkingTab(ttk.Frame):
    VEHICLE_NAMES = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike",
        "vehicle",
        "van",
        "auto",
        "autorickshaw",
        "rickshaw",
        "scooter",
    }

    def __init__(self, app, parent):
        super().__init__(parent)
        self.app = app

        self.model = None
        self.model_path = None
        self.model_names = {}
        self.running = False

        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value="")
        self.json_path_var = tk.StringVar(value="")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.iou_var = tk.DoubleVar(value=DEFAULT_IOU)
        self.overlap_var = tk.DoubleVar(value=DEFAULT_OVERLAP)
        self.vehicle_class_var = tk.StringVar(value="All vehicle-like classes")

        self._build()
        # Pass the actual ParkingTab video canvas to the ROI editor.
        self.editor = ParkingEditor(app, self.video)

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        # Source controls.
        source = ttk.LabelFrame(top, text="Parking Video", padding=6)
        source.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ttk.Button(
            source,
            text="Select Video",
            command=self.select_video,
        ).pack(side="left")

        ttk.Label(
            source,
            textvariable=self.video_path_var,
            width=45,
        ).pack(side="left", padx=8)

        # Model controls.
        model = ttk.LabelFrame(top, text="Vehicle YOLO Model", padding=6)
        model.pack(side="left", fill="x", expand=True)

        ttk.Button(
            model,
            text="Select .pt",
            command=self.select_model,
        ).pack(side="left")

        ttk.Label(
            model,
            textvariable=self.model_path_var,
            width=38,
        ).pack(side="left", padx=8)

        settings = ttk.Frame(self, padding=(8, 0, 8, 5))
        settings.pack(fill="x")

        editor_box = ttk.LabelFrame(
            settings,
            text="Parking Slot Editor",
            padding=6,
        )
        editor_box.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ttk.Button(
            editor_box,
            text="➕ Add Parking Slot",
            command=self.app.parking_editor_add,
        ).pack(side="left", padx=2)

        ttk.Button(
            editor_box,
            text="🗑 Delete Selected",
            command=self.app.parking_editor_delete,
        ).pack(side="left", padx=2)

        ttk.Button(
            editor_box,
            text="Clear",
            command=self.app.parking_editor_clear,
        ).pack(side="left", padx=2)

        ttk.Button(
            editor_box,
            text="Save JSON",
            command=self.app.parking_editor_save,
        ).pack(side="left", padx=2)

        ttk.Button(
            editor_box,
            text="Load JSON",
            command=self.app.parking_editor_load,
        ).pack(side="left", padx=2)

        ttk.Label(editor_box, text="Edge:").pack(side="left", padx=(10, 2))
        ttk.Spinbox(
            editor_box,
            from_=1,
            to=20,
            width=4,
            textvariable=self.app.parking_editor_width_var,
            command=self.app.parking_editor_width_changed,
        ).pack(side="left")

        ttk.Button(
            editor_box,
            text="Color",
            command=self.app.parking_editor_color,
        ).pack(side="left", padx=4)

        infer_box = ttk.LabelFrame(
            settings,
            text="Occupancy",
            padding=6,
        )
        infer_box.pack(side="left", fill="x")

        ttk.Label(infer_box, text="Confidence").pack(side="left")
        ttk.Spinbox(
            infer_box,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=5,
            textvariable=self.conf_var,
        ).pack(side="left", padx=(3, 8))

        ttk.Label(infer_box, text="IoU").pack(side="left")
        ttk.Spinbox(
            infer_box,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=5,
            textvariable=self.iou_var,
        ).pack(side="left", padx=(3, 8))

        ttk.Label(infer_box, text="Slot overlap").pack(side="left")
        ttk.Spinbox(
            infer_box,
            from_=0.05,
            to=1.0,
            increment=0.05,
            width=5,
            textvariable=self.overlap_var,
        ).pack(side="left", padx=(3, 8))

        ttk.Label(infer_box, text="Vehicle classes").pack(side="left")
        self.vehicle_combo = ttk.Combobox(
            infer_box,
            textvariable=self.vehicle_class_var,
            state="readonly",
            width=24,
        )
        self.vehicle_combo.pack(side="left", padx=3)

        main = ttk.Frame(self, padding=(8, 0, 8, 8))
        main.pack(fill="both", expand=True)

        self.video = VideoCanvas(main)
        self.video.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(main, width=250)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        self.summary_var = tk.StringVar(
            value="TOTAL: 0   FREE: 0   OCCUPIED: 0"
        )
        ttk.Label(
            side,
            textvariable=self.summary_var,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(5, 10))

        ttk.Label(
            side,
            text="Parking Slots",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")

        self.slot_list = tk.Listbox(side, height=12)
        self.slot_list.pack(fill="x", pady=5)
        self.slot_list.bind("<<ListboxSelect>>", self._slot_selected)
        self.slot_context_menu = tk.Menu(self.slot_list, tearoff=0)
        self.slot_context_menu.add_command(
            label="Duplicate",
            command=self.app.parking_editor_duplicate,
        )
        self.slot_context_menu.add_command(
            label="Delete",
            command=self.app.parking_editor_delete,
        )
        for sequence in _right_click_bindings(self.slot_list):
            self.slot_list.bind(sequence, self._show_slot_context_menu)

        self.status_var = tk.StringVar(value="Select a video.")
        ttk.Label(
            side,
            textvariable=self.status_var,
            wraplength=230,
        ).pack(anchor="w", pady=5)

        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="⏮ Previous",
            command=self.app.parking_previous,
        ).pack(side="left", padx=2)

        self.play_btn = ttk.Button(
            controls,
            text="▶ Run YOLO",
            command=self.run_yolo,
        )
        self.play_btn.pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="⏸ Pause",
            command=self.app.parking_pause,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="⏹ Stop",
            command=self.app.parking_stop,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="Next ⏭",
            command=self.app.parking_next,
        ).pack(side="left", padx=2)

        ttk.Label(controls, text="Speed:").pack(side="left", padx=(12, 3))
        self.speed_combo = ttk.Combobox(
            controls,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            state="readonly",
            width=7,
        )
        self.speed_combo.set("1.0x")
        self.speed_combo.pack(side="left")
        self.speed_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.app.parking_set_speed(
                float(self.speed_combo.get().replace("x", ""))
            ),
        )

        self.timeline = ttk.Scale(
            controls,
            from_=0,
            to=1,
            orient="horizontal",
            command=lambda _v: None,
        )
        self.timeline.pack(side="left", fill="x", expand=True, padx=10)

        self.video.timeline = self.timeline

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select Parking Video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.video_path_var.set(path)
            self.app.open_parking_video(path)
            self.update_slot_list()
        except Exception as exc:
            messagebox.showerror("Video error", str(exc))

    def select_model(self):
        path = filedialog.askopenfilename(
            title="Select Vehicle YOLO Model",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        self.load_model(path)

    def load_model(self, path):
        self.model = None
        self.model_path = None
        self.model_path_var.set("")
        self.names = {}
        self.class_combo["values"] = ["All classes"]
        self.class_var.set("All classes")

        if YOLO is None:
            messagebox.showerror(
                "Ultralytics missing",
                "Install dependencies first:\n\npip install -r requirements.txt",
            )
            return

        try:
            self.status_var.set("Loading YOLO model...")
            self.update_idletasks()

            model = YOLO(path)
            self.model = model
            self.model_path = path
            self.model_path_var.set(path)
            self.model_names = model_names(model)

            options = ["All vehicle-like classes"]
            options += list(self.model_names.values())
            self.vehicle_combo["values"] = options

            # Prefer common vehicle names when present.
            common = [
                name
                for name in self.model_names.values()
                if name.lower() in self.VEHICLE_NAMES
            ]
            if common:
                self.vehicle_class_var.set(common[0])
            else:
                self.vehicle_class_var.set("All vehicle-like classes")

            self.status_var.set(
                f"Model loaded: {Path(path).name}\n"
                f"Classes: {len(self.model_names)}\n"
                f"Device: {choose_device()}"
            )
        except Exception as exc:
            self.model = None
            messagebox.showerror(
                "Model error",
                f"Could not load model:\n{exc}",
            )

    def update_slot_list(self, occupancy=None):
        self.slot_list.delete(0, tk.END)
        for slot in self.app.parking_editor.slots:
            state = ""
            if occupancy and slot["id"] in occupancy:
                state = "  -  " + occupancy[slot["id"]]
            self.slot_list.insert(
                tk.END,
                f'{slot["id"]} - Parking Slot{state}',
            )

    def _slot_selected(self, _event):
        selection = self.slot_list.curselection()
        if not selection:
            return
        self.app.parking_editor.select_slot(selection[0])

    def _show_slot_context_menu(self, event):
        index = self.slot_list.nearest(event.y)
        row_bounds = self.slot_list.bbox(index)
        if (
            not 0 <= index < self.slot_list.size()
            or row_bounds is None
            or not row_bounds[1] <= event.y < row_bounds[1] + row_bounds[3]
        ):
            return "break"

        self.slot_list.selection_clear(0, tk.END)
        self.slot_list.selection_set(index)
        self.app.parking_editor.select_slot(index)
        self.slot_context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def run_yolo(self):
        if self.model is None:
            messagebox.showwarning(
                "Model required",
                "Select a vehicle YOLO .pt model first.",
            )
            return

        if self.video.source.cap is None:
            messagebox.showwarning(
                "Video required",
                "Select a parking video first.",
            )
            return

        if not self.app.parking_editor.slots:
            messagebox.showwarning(
                "Parking slots required",
                "Add at least one Parking Slot polygon first.",
            )
            return

        self.running = True
        self.play_btn.configure(text="▶ Running...")
        self.app.parking_start_inference()

    def infer_frame(self, frame, _index):
        if self.model is None:
            return frame

        device = choose_device()
        conf = max(0.01, min(0.99, float(self.conf_var.get())))
        iou = max(0.01, min(0.99, float(self.iou_var.get())))
        overlap_threshold = max(
            0.01,
            min(1.0, float(self.overlap_var.get())),
        )

        selected_class = self.vehicle_class_var.get()

        kwargs = {
            "source": frame,
            "conf": conf,
            "iou": iou,
            "verbose": False,
            "device": device,
        }

        result = self.model.predict(**kwargs)[0]
        boxes = result.boxes

        detections = []
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            scores = boxes.conf.detach().cpu().numpy()

            for box, c, score in zip(xyxy, cls, scores):
                name = self.model_names.get(int(c), str(c))

                if (
                    selected_class != "All vehicle-like classes"
                    and name != selected_class
                ):
                    continue

                if selected_class == "All vehicle-like classes":
                    # If the model contains known vehicle classes, use them.
                    # If it is a custom model with unknown class names, allow
                    # all detections so the workflow still works.
                    known_vehicle_names = {
                        n.lower() for n in self.model_names.values()
                    } & self.VEHICLE_NAMES
                    if known_vehicle_names and name.lower() not in known_vehicle_names:
                        continue

                detections.append((box, name, float(score)))

        occupancy = {}
        for slot in self.app.parking_editor.slots:
            occupied = False
            best_overlap = 0.0

            for box, _name, _score in detections:
                ratio = polygon_intersection_ratio(
                    slot["points"],
                    box,
                )
                best_overlap = max(best_overlap, ratio)

                if ratio >= overlap_threshold:
                    occupied = True
                    break

            occupancy[slot["id"]] = "OCCUPIED" if occupied else "FREE"

        output = frame.copy()

        # Draw vehicle detections.
        for box, name, score in detections:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 200, 0),
                2,
            )
            label = f"{name} {score:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw parking slots and states.
        free_count = 0
        occupied_count = 0

        for slot in self.app.parking_editor.slots:
            state = occupancy.get(slot["id"], "FREE")
            if state == "OCCUPIED":
                color = (0, 0, 255)
                occupied_count += 1
            else:
                color = (0, 255, 0)
                free_count += 1

            pts = np.round(
                np.asarray(slot["points"], dtype=np.float32)
            ).astype(np.int32).reshape((-1, 1, 2))

            width = max(2, int(slot.get("edge_width", 3)))
            cv2.polylines(
                output,
                [pts],
                True,
                color,
                width,
                cv2.LINE_AA,
            )

            x, y = int(slot["points"][0][0]), int(slot["points"][0][1])
            cv2.putText(
                output,
                f'{slot["id"]} - {state}',
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        total = len(self.app.parking_editor.slots)
        cv2.rectangle(output, (10, 10), (330, 78), (20, 20, 20), -1)
        cv2.putText(
            output,
            f"TOTAL: {total}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"FREE: {free_count}   OCCUPIED: {occupied_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.app.after(
            0,
            lambda o=occupancy: self._update_occupancy_ui(o),
        )

        return output

    def _update_occupancy_ui(self, occupancy):
        self.summary_var.set(
            f"TOTAL: {len(self.app.parking_editor.slots)}   "
            f"FREE: {sum(v == 'FREE' for v in occupancy.values())}   "
            f"OCCUPIED: {sum(v == 'OCCUPIED' for v in occupancy.values())}"
        )
        self.update_slot_list(occupancy)

    def plain_frame(self, frame, _index):
        output = frame.copy()
        self.app.parking_editor.draw_overlays(output)
        return output


class YOLOTestTab(ttk.Frame):
    """Interactive YOLO playback and saved single/comparison test workflow."""

    def __init__(self, app, parent):
        super().__init__(parent)
        self.app = app

        self.model = None
        self.model_path = None
        self.names = {}
        self.compare_model_path = None
        self.test_queue = queue.Queue()
        self.test_thread = None

        self.model_path_var = tk.StringVar(value="")
        self.video_path_var = tk.StringVar(value="")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.iou_var = tk.DoubleVar(value=DEFAULT_IOU)
        self.class_var = tk.StringVar(value="All classes")
        self.test_name_var = tk.StringVar(value="")
        self.test_mode_var = tk.StringVar(value="Test one model")
        self.compare_model_var = tk.StringVar(value="No comparison model selected")
        self.status_var = tk.StringVar(value="Select a model and video.")

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Button(
            top,
            text="Select YOLO .pt Model",
            command=self.select_model,
        ).pack(side="left")

        ttk.Label(
            top,
            textvariable=self.model_path_var,
            width=45,
        ).pack(side="left", padx=8)

        ttk.Button(
            top,
            text="Select Test Video",
            command=self.select_video,
        ).pack(side="left", padx=(15, 0))

        ttk.Label(
            top,
            textvariable=self.video_path_var,
            width=42,
        ).pack(side="left", padx=8)

        settings = ttk.Frame(self, padding=(8, 0, 8, 5))
        settings.pack(fill="x")

        ttk.Label(settings, text="Class:").pack(side="left")
        self.class_combo = ttk.Combobox(
            settings,
            textvariable=self.class_var,
            state="readonly",
            width=25,
            values=["All classes"],
        )
        self.class_combo.pack(side="left", padx=(4, 12))

        ttk.Label(settings, text="Confidence:").pack(side="left")
        ttk.Spinbox(
            settings,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=6,
            textvariable=self.conf_var,
        ).pack(side="left", padx=(4, 12))

        ttk.Label(settings, text="IoU:").pack(side="left")
        ttk.Spinbox(
            settings,
            from_=0.05,
            to=0.95,
            increment=0.05,
            width=6,
            textvariable=self.iou_var,
        ).pack(side="left", padx=4)

        test_settings = ttk.LabelFrame(self, text="Saved model test", padding=6)
        test_settings.pack(fill="x", padx=8, pady=(0, 6))

        ttk.Label(test_settings, text="Test name:").pack(side="left")
        ttk.Entry(
            test_settings,
            textvariable=self.test_name_var,
            width=20,
        ).pack(side="left", padx=(4, 10))

        self.test_mode_combo = ttk.Combobox(
            test_settings,
            textvariable=self.test_mode_var,
            state="readonly",
            values=["Test one model", "Compare two models"],
            width=20,
        )
        self.test_mode_combo.pack(side="left", padx=3)
        self.test_mode_combo.bind(
            "<<ComboboxSelected>>",
            self._test_mode_changed,
        )

        self.compare_model_button = ttk.Button(
            test_settings,
            text="Select second model",
            command=self.select_compare_model,
        )
        self.compare_model_button.pack(side="left", padx=4)

        ttk.Label(
            test_settings,
            textvariable=self.compare_model_var,
            width=24,
        ).pack(side="left", padx=4)

        self.saved_test_button = ttk.Button(
            test_settings,
            text="▶ Run and save test",
            command=self.run_saved_test,
        )
        self.saved_test_button.pack(side="left", padx=(8, 4))
        ttk.Button(
            test_settings,
            text="Previous tests",
            command=self.show_test_history,
        ).pack(side="left", padx=4)
        self._test_mode_changed()

        main = ttk.Frame(self, padding=(8, 0, 8, 8))
        main.pack(fill="both", expand=True)

        self.video = VideoCanvas(main)
        self.video.pack(fill="both", expand=True)

        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="⏮ Previous",
            command=self.previous,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="▶ Run Model",
            command=self.run_model,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="⏸ Pause",
            command=self.pause,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="⏹ Stop",
            command=self.stop,
        ).pack(side="left", padx=2)

        ttk.Button(
            controls,
            text="Next ⏭",
            command=self.next,
        ).pack(side="left", padx=2)

        ttk.Label(controls, text="Speed:").pack(side="left", padx=(12, 3))
        self.speed_combo = ttk.Combobox(
            controls,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            state="readonly",
            width=7,
        )
        self.speed_combo.set("1.0x")
        self.speed_combo.pack(side="left")
        self.speed_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.app.yolo_set_speed(
                float(self.speed_combo.get().replace("x", ""))
            ),
        )

        self.timeline = ttk.Scale(
            controls,
            from_=0,
            to=1,
            orient="horizontal",
        )
        self.timeline.pack(side="left", fill="x", expand=True, padx=10)

        self.status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(8, 0, 8, 8),
        )
        self.status_label.pack(fill="x")

    def select_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO .pt model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        self.load_model(path)

    def _test_mode_changed(self, _event=None):
        comparing = self.test_mode_var.get() == "Compare two models"
        state = "normal" if comparing else "disabled"
        self.compare_model_button.configure(state=state)
        if not comparing:
            self.compare_model_path = None
            self.compare_model_var.set("No comparison model selected")

    def select_compare_model(self):
        path = filedialog.askopenfilename(
            title="Select second YOLO Model",
            filetypes=[("YOLO .pt model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.compare_model_path = path
        self.compare_model_var.set(Path(path).name)

    def load_model(self, path):
        if YOLO is None:
            messagebox.showerror(
                "Ultralytics missing",
                "Install dependencies:\n\npip install -r requirements.txt",
            )
            return

        try:
            self.status_var.set("Loading model...")
            self.update_idletasks()

            self.model = YOLO(path)
            self.model_path = path
            self.model_path_var.set(path)
            self.names = model_names(self.model)

            values = ["All classes"] + list(self.names.values())
            self.class_combo["values"] = values
            self.class_var.set("All classes")

            self.status_var.set(
                f"Loaded {Path(path).name} | "
                f"{len(self.names)} classes | Device: {choose_device()}"
            )
        except Exception as exc:
            self.status_var.set("Could not load model.")
            messagebox.showerror(
                "Model error",
                f"Could not load YOLO model:\n{exc}",
            )

    def run_saved_test(self):
        if self.test_thread is not None and self.test_thread.is_alive():
            messagebox.showwarning(
                "Test in progress",
                "Wait for the current saved test to finish.",
            )
            return
        if self.app.yolo_controller.worker is not None and (
            self.app.yolo_controller.worker.is_alive()
        ):
            messagebox.showwarning(
                "Playback in progress",
                "Stop model playback and wait for it to finish before "
                "starting a saved test.",
            )
            return

        test_name = self.test_name_var.get().strip()
        if not test_name:
            messagebox.showwarning("Test name required", "Enter a test name.")
            return
        if not self.model_path:
            messagebox.showwarning(
                "Model required",
                "Select the first YOLO model before running a test.",
            )
            return
        video_path = self.video_path_var.get()
        if not video_path or not Path(video_path).is_file():
            messagebox.showwarning(
                "Video required",
                "Select a valid test video before running a test.",
            )
            return

        model_paths = [self.model_path]
        if self.test_mode_var.get() == "Compare two models":
            if not self.compare_model_path:
                messagebox.showwarning(
                    "Second model required",
                    "Select the second YOLO model for comparison.",
                )
                return
            if Path(self.model_path).resolve() == Path(
                self.compare_model_path
            ).resolve():
                messagebox.showwarning(
                    "Different models required",
                    "Select two different model files to compare.",
                )
                return
            model_paths.append(self.compare_model_path)

        safe_name = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in test_name
        ).strip("_")
        if not safe_name:
            messagebox.showwarning(
                "Invalid test name",
                "The test name must contain at least one letter or number.",
            )
            return

        try:
            confidence = max(0.01, min(0.99, float(self.conf_var.get())))
            iou = max(0.01, min(0.99, float(self.iou_var.get())))
        except (tk.TclError, ValueError) as exc:
            messagebox.showwarning(
                "Invalid test settings",
                f"Confidence and IoU must be valid numbers.\n\n{exc}",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        test_folder = RESULTS_DIR / f"{safe_name}_{timestamp}"
        config = {
            "test_name": test_name,
            "video_path": video_path,
            "model_paths": model_paths,
            "confidence": confidence,
            "iou": iou,
            "test_folder": test_folder,
        }

        self.app.yolo_controller.pause()
        self.saved_test_button.configure(state="disabled")
        self.status_var.set("Preparing saved model test...")
        self.test_thread = threading.Thread(
            target=self._saved_test_worker,
            args=(config,),
            daemon=True,
        )
        self.test_thread.start()
        self.after(100, self._poll_saved_test)

    def _saved_test_worker(self, config):
        try:
            test_folder = config["test_folder"]
            test_folder.mkdir(parents=True, exist_ok=False)
            source_video = Path(config["video_path"])
            input_video = test_folder / f"input_{source_video.name}"
            shutil.copy2(source_video, input_video)
            video_info = video_information(input_video)

            model_results = []
            for index, model_path in enumerate(config["model_paths"], start=1):
                self.test_queue.put(
                    ("status", f"Running model {index}/{len(config['model_paths'])}: "
                     f"{Path(model_path).name}")
                )
                output_folder = (
                    test_folder
                    if len(config["model_paths"]) == 1
                    else test_folder / f"model_{index}"
                )
                model_results.append(
                    run_saved_model_test(
                        model_path,
                        input_video,
                        output_folder,
                        config["confidence"],
                        config["iou"],
                    )
                )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if len(model_results) == 1:
                result = model_results[0]
                data = {
                    "test_name": config["test_name"],
                    "model": result["name"],
                    "video": source_video.name,
                    "confidence": config["confidence"],
                    "iou": config["iou"],
                    "video_information": video_info,
                    "total_detections": result["total_detections"],
                    "class_detections": result["class_counts"],
                    "processing_time_seconds": result["processing_time"],
                    "output_video": result["output_video"],
                    "date": timestamp,
                }
                metadata_name = "test_info.json"
            else:
                data = {
                    "test_name": config["test_name"],
                    "model_1": model_results[0],
                    "model_2": model_results[1],
                    "confidence": config["confidence"],
                    "iou": config["iou"],
                    "video": video_info,
                    "video_name": source_video.name,
                    "date": timestamp,
                }
                metadata_name = "comparison.json"

            with (test_folder / metadata_name).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(data, file, indent=4)

            self.test_queue.put(("complete", (test_folder, data)))
        except Exception as exc:
            self.test_queue.put(("error", str(exc)))

    def _poll_saved_test(self):
        try:
            while True:
                kind, payload = self.test_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.saved_test_button.configure(state="normal")
                    self.status_var.set("Saved test failed.")
                    messagebox.showerror("Saved test failed", payload)
                elif kind == "complete":
                    self.saved_test_button.configure(state="normal")
                    test_folder, data = payload
                    self.status_var.set(f"Saved test: {test_folder}")
                    self.show_test_report(test_folder, data)
        except queue.Empty:
            pass

        if self.test_thread is not None and self.test_thread.is_alive():
            self.after(100, self._poll_saved_test)
        else:
            self.saved_test_button.configure(state="normal")

    def show_test_history(self):
        window = tk.Toplevel(self)
        window.title("Previous YOLO Tests")
        window.geometry("850x400")

        columns = ("test", "mode", "date", "summary")
        table = ttk.Treeview(window, columns=columns, show="headings")
        for column, label in zip(
            columns,
            ("Test name", "Mode", "Date", "Summary"),
        ):
            table.heading(column, text=label)
        table.column("test", width=190)
        table.column("mode", width=110)
        table.column("date", width=150)
        table.column("summary", width=350)
        table.pack(fill="both", expand=True, padx=8, pady=8)

        folders = {}
        if RESULTS_DIR.exists():
            for folder in sorted(
                (path for path in RESULTS_DIR.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:50]:
                metadata = folder / "test_info.json"
                if not metadata.exists():
                    metadata = folder / "comparison.json"
                try:
                    with metadata.open(encoding="utf-8") as file:
                        data = json.load(file)
                except (OSError, json.JSONDecodeError) as exc:
                    table.insert(
                        "",
                        "end",
                        iid=folder.name,
                        values=(folder.name, "Unreadable", "", str(exc)),
                    )
                    folders[folder.name] = folder
                    continue

                comparing = "model_1" in data and "model_2" in data
                if comparing:
                    summary = (
                        f'{data["model_1"]["name"]} vs '
                        f'{data["model_2"]["name"]}'
                    )
                else:
                    summary = (
                        f'{data.get("total_detections", 0)} detections | '
                        f'{data.get("model", "Unknown model")}'
                    )
                table.insert(
                    "",
                    "end",
                    iid=folder.name,
                    values=(
                        data.get("test_name", folder.name),
                        "Comparison" if comparing else "Single model",
                        data.get("date", ""),
                        summary,
                    ),
                )
                folders[folder.name] = folder

        def open_selected(_event=None):
            selection = table.selection()
            if not selection:
                return
            folder = folders[selection[0]]
            metadata = folder / "test_info.json"
            if not metadata.exists():
                metadata = folder / "comparison.json"
            try:
                with metadata.open(encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                messagebox.showerror(
                    "History error",
                    f"Could not read test results:\n{exc}",
                    parent=window,
                )
                return
            self.show_test_report(folder, data)

        table.bind("<Double-1>", open_selected)
        ttk.Button(
            window,
            text="Open selected test",
            command=open_selected,
        ).pack(pady=(0, 8))
        if not folders:
            ttk.Label(window, text="No previous tests found.").pack(pady=8)

    def show_test_report(self, folder, data):
        window = tk.Toplevel(self)
        window.title(f'Test results - {data.get("test_name", folder.name)}')
        window.geometry("680x480")

        details = tk.Text(window, wrap="word", height=18)
        details.pack(fill="both", expand=True, padx=8, pady=8)
        details.insert("end", f'Test: {data.get("test_name", folder.name)}\n')
        details.insert("end", f'Date: {data.get("date", "Unknown")}\n')
        details.insert("end", f'Video: {data.get("video_name", data.get("video", "Unknown"))}\n')
        details.insert("end", f'Confidence: {data.get("confidence", "Unknown")}\n')
        details.insert("end", f'IoU: {data.get("iou", "Unknown")}\n')

        models = (
            [data["model_1"], data["model_2"]]
            if "model_1" in data and "model_2" in data
            else [{
                "name": data.get("model", "Unknown model"),
                "total_detections": data.get("total_detections", 0),
                "class_counts": data.get("class_detections", {}),
                "processing_time": data.get(
                    "processing_time_seconds",
                    0,
                ),
                "output_video": data.get("output_video"),
            }]
        )
        for index, result in enumerate(models, start=1):
            title = f'\nModel {index}: {result.get("name", "Unknown")}\n'
            if len(models) == 1:
                title = f'\nModel: {result.get("name", "Unknown")}\n'
            details.insert("end", title)
            details.insert(
                "end",
                f'Detections: {result.get("total_detections", 0)}\n'
                f'Processing time: {result.get("processing_time", 0)} s\n'
                f'Class counts: {result.get("class_counts", {})}\n',
            )

        details.insert("end", f"\nResults folder: {folder}\n")
        details.configure(state="disabled")

        outputs = ttk.Frame(window, padding=(8, 0, 8, 8))
        outputs.pack(fill="x")
        available_outputs = 0
        for index, result in enumerate(models, start=1):
            relative_path = result.get("output_video")
            if not relative_path:
                continue
            output_path = folder / relative_path
            if not output_path.is_file():
                continue
            available_outputs += 1
            label = f'Load Model {index} output' if len(models) > 1 else "Load result video"
            ttk.Button(
                outputs,
                text=label,
                command=lambda path=output_path: self._load_result_video(path),
            ).pack(side="left", padx=4)
        if not available_outputs:
            ttk.Label(outputs, text="No annotated output video was generated.").pack(
                side="left",
                padx=4,
            )

    def _load_result_video(self, path):
        try:
            self.app.open_yolo_video(str(path))
        except Exception as exc:
            messagebox.showerror("Result video error", str(exc))

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Test Video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.video_path_var.set(path)
            self.app.open_yolo_video(path)
        except Exception as exc:
            messagebox.showerror("Video error", str(exc))

    def run_model(self):
        if self.model is None:
            messagebox.showwarning(
                "Model required",
                "Select a trained/retrained .pt model first.",
            )
            return

        if self.video.source.cap is None:
            messagebox.showwarning(
                "Video required",
                "Select a test video first.",
            )
            return

        self.app.yolo_start_inference()

    def infer_frame(self, frame, _index):
        if self.model is None:
            return frame

        conf = max(0.01, min(0.99, float(self.conf_var.get())))
        iou = max(0.01, min(0.99, float(self.iou_var.get())))
        selected = self.class_var.get()

        result = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=choose_device(),
        )[0]

        output = frame.copy()
        boxes = result.boxes

        count = 0

        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            scores = boxes.conf.detach().cpu().numpy()

            for box, c, score in zip(xyxy, cls, scores):
                name = self.names.get(int(c), str(c))

                if selected != "All classes" and name != selected:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                label = f"{name} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2,
                )

                cv2.rectangle(
                    output,
                    (x1, max(0, y1 - th - 8)),
                    (x1 + tw + 6, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    output,
                    label,
                    (x1 + 3, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                count += 1

        cv2.putText(
            output,
            f"Detections: {count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    def previous(self):
        self.app.yolo_previous()

    def next(self):
        self.app.yolo_next()

    def pause(self):
        self.app.yolo_pause()

    def stop(self):
        self.app.yolo_stop()


class AIAnalyticsApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1450x900")
        self.minsize(1100, 700)

        self.parking_json_path = tk.StringVar(value="")
        self.parking_editor_width_var = tk.IntVar(
            value=DEFAULT_EDGE_WIDTH
        )

        self.parking_source = VideoSource()
        self.yolo_source = VideoSource()

        self._build_style()
        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Title.TLabel",
            font=("TkDefaultFont", 15, "bold"),
        )

        style.configure(
            "Big.TButton",
            padding=(10, 7),
        )

    def _build_ui(self):
        header = ttk.Frame(self, padding=(10, 8))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="AI Video Analytics",
            style="Title.TLabel",
        ).pack(side="left")

        ttk.Label(
            header,
            text="   Parking Analytics + Generic YOLO Model Testing",
        ).pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        parking_frame = ttk.Frame(self.notebook)
        yolo_frame = ttk.Frame(self.notebook)

        self.notebook.add(
            parking_frame,
            text="🅿 Parking Analytics",
        )
        self.notebook.add(
            yolo_frame,
            text="🤖 YOLO Model Test",
        )

        # Create parking tab first.
        self.parking_tab = ParkingTab(self, parking_frame)
        self.parking_tab.pack(fill="both", expand=True)

        # ParkingEditor is created by ParkingTab after its VideoCanvas exists.
        self.parking_editor = self.parking_tab.editor

        self.yolo_tab = YOLOTestTab(self, yolo_frame)
        self.yolo_tab.pack(fill="both", expand=True)

        # The ParkingTab's VideoCanvas must be connected to the controller.
        self.parking_controller = VideoController(
            self,
            self.parking_tab.video,
            self.parking_tab.timeline,
            self.parking_tab.status_var,
        )

        self.yolo_controller = VideoController(
            self,
            self.yolo_tab.video,
            self.yolo_tab.timeline,
            self.yolo_tab.status_var,
        )

    # ---------------------------
    # Parking workflow
    # ---------------------------

    def open_parking_video(self, path):
        self.parking_controller.open(path)

        # New source may have a different resolution. Existing slots are not
        # silently altered here; JSON loading performs explicit scaling.
        self.parking_editor.redraw()

    def parking_editor_add(self):
        self.parking_editor.add_slot()
        self.parking_tab.update_slot_list()

    def parking_editor_delete(self):
        self.parking_editor.delete_selected()
        self.parking_tab.update_slot_list()

    def parking_editor_duplicate(self):
        self.parking_editor.duplicate_selected()
        self.parking_tab.update_slot_list()
        if self.parking_editor.selected is not None:
            self.parking_tab.slot_list.selection_set(
                self.parking_editor.selected
            )
            self.parking_tab.slot_list.activate(
                self.parking_editor.selected
            )

    def parking_editor_clear(self):
        self.parking_editor.clear()
        self.parking_tab.update_slot_list()
        self.parking_tab.summary_var.set(
            "TOTAL: 0   FREE: 0   OCCUPIED: 0"
        )

    def parking_editor_save(self):
        self.parking_editor.save_json()

    def parking_editor_load(self):
        if self.parking_controller.source.cap is None:
            messagebox.showwarning(
                "Video required",
                "Select the parking video before loading its ROI JSON.",
            )
            return
        self.parking_editor.load_json()
        self.parking_tab.update_slot_list()

    def parking_editor_width_changed(self):
        self.parking_editor.edge_width_var.set(
            self.parking_editor_width_var.get()
        )
        self.parking_editor.set_edge_width()

    def parking_editor_color(self):
        self.parking_editor.choose_color()

    def parking_start_inference(self):
        self.parking_controller.on_frame = None
        self.parking_controller.play(
            infer_function=self.parking_tab.infer_frame,
            mode_name="parking",
        )

    def parking_pause(self):
        self.parking_controller.pause()

    def parking_stop(self):
        self.parking_controller.stop()
        self.parking_tab.running = False
        self.parking_tab.play_btn.configure(text="▶ Run YOLO")

    def parking_previous(self):
        self.parking_controller.previous()

    def parking_next(self):
        self.parking_controller.next_frame()

    def parking_set_speed(self, speed):
        self.parking_controller.set_speed(speed)

    # ---------------------------
    # Generic YOLO workflow
    # ---------------------------

    def open_yolo_video(self, path):
        self.yolo_controller.open(path)

    def yolo_start_inference(self):
        self.yolo_controller.play(
            infer_function=self.yolo_tab.infer_frame,
            mode_name="yolo",
        )

    def yolo_pause(self):
        self.yolo_controller.pause()

    def yolo_stop(self):
        self.yolo_controller.stop()

    def yolo_previous(self):
        self.yolo_controller.previous()

    def yolo_next(self):
        self.yolo_controller.next_frame()

    def yolo_set_speed(self, speed):
        self.yolo_controller.set_speed(speed)

    # ---------------------------
    # Errors / cleanup
    # ---------------------------

    def handle_worker_error(self, error):
        self.parking_controller.pause()
        self.yolo_controller.pause()
        messagebox.showerror(
            "Inference error",
            f"YOLO inference stopped:\n\n{error}",
        )

    def on_close(self):
        try:
            self.parking_controller.stop()
            self.yolo_controller.stop()
        except Exception:
            pass

        try:
            self.parking_source.release()
            self.yolo_source.release()
        except Exception:
            pass

        self.destroy()


def main():
    app = AIAnalyticsApp()

    # ParkingEditor references are ready after UI construction.
    app.parking_editor_width_var.set(
        app.parking_editor.edge_width_var.get()
    )

    app.mainloop()


if __name__ == "__main__":
    main()
