import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from queue import Queue

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None

try:
    import easyocr
except Exception:
    easyocr = None

from fire_logic import FireAlertDetector
from logic_manager import resolve_selected_logic, sync_logic_checkbox_group
from counting_logic import (
    LineCrossingCounter,
    draw_count_summary_overlay,
    draw_counting_line,
    summarize_model_counts,
)

MAX_PREVIEW_WIDTH = 720
MAX_PREVIEW_HEIGHT = 405

class DetectionGUI:
    def __init__(self):
        self.models = []  # List of loaded models
        self.active_model_idx = 0  # Index of currently active model
        self.selected_classes = set()  # Store selected class indices
        self.class_names = {}  # Will be updated when model is loaded
        self.unified_class_names = {}  # name -> idx
        self.writer = None  # Initialize video writer
        self.writer_size = None
        
        # Custom class colors: class_name -> BGR color tuple
        self.custom_class_colors = {
            'TACTILE-TILES': (0, 255, 0),  # Green in BGR
        }
        
        # Batch processing state
        self.batch_processing = False
        self.batch_executor = None
        self.batch_tasks = {}
        self.batch_progress_queue = Queue()
        self.batch_preview_frames = {}  # Dict to store frames from each video
        self.batch_preview_lock = threading.Lock()
        self.active_batch_videos = []  # Track active video names for grid
        self.batch_current_video_idx = 0  # For slideshow rotation
        self.batch_video_names = []  # Track video names for slideshow
        self._batch_settings = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("YOLOv8 Detection")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(1, min(1600, screen_width - 48))
        window_height = max(1, min(900, screen_height - 96))
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.compare_var = tk.BooleanVar(master=self.root, value=False)
        self.mix_var = tk.BooleanVar(master=self.root, value=False)
        # BGR colors used to tell the two mixed models apart on one frame
        self.mix_colors_bgr = [(0, 220, 0), (0, 140, 255)]
        self.mix_colors_hex = ["#00DC00", "#FF8C00"]
        self.is_live_source = False
        self.current_video_path = None
        self._last_raw_frame = None
        self.output_video_path = os.path.join(os.getcwd(), 'output_segmentation.mp4')
        self.fire_detector = FireAlertDetector(
            model_path=os.path.join(os.getcwd(), 'fire_model.pt'),
            conf_threshold=0.70,
        )
        self.fire_alert_active = False
        self.last_fire_alert_time = 0.0
        self.fire_logic_mode = "Fire Alert + Sound"
        self.count_smoothing_window = 5
        self.count_history = []
        self.line_counter = LineCrossingCounter()
        self.plate_text = ""
        self.plate_text_box = None
        self.ocr_reader = None
        if easyocr is not None:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception:
                self.ocr_reader = None
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.play_pause())
        self.root.bind('<Left>', lambda e: self.skip_backward())
        self.root.bind('<Right>', lambda e: self.skip_forward())
        self.root.focus_set()  # Allow keyboard focus
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create model management frame at the top
        model_frame = ttk.LabelFrame(main_container, text="Model Management")
        model_frame.pack(fill=tk.X, padx=3, pady=2)
        
        # Model selection buttons and loaded models display
        model_toolbar = ttk.Frame(model_frame)
        model_toolbar.pack(fill=tk.X, padx=3, pady=0)
        self.model_buttons_frame = ttk.Frame(model_toolbar)
        self.model_buttons_frame.pack(side=tk.LEFT, padx=2)
        ttk.Button(self.model_buttons_frame, text="Add Model", command=self.add_model).pack(side=tk.LEFT, padx=2, pady=1)
        self.compare_checkbox = ttk.Checkbutton(model_toolbar, text="Compare Models", variable=self.compare_var, command=self.toggle_comparison)
        self.compare_checkbox.pack(side=tk.LEFT, padx=3, pady=1)
        self.mix_checkbox = ttk.Checkbutton(model_toolbar, text="Mix Models", variable=self.mix_var, command=self.toggle_mix)
        self.mix_checkbox.pack(side=tk.LEFT, padx=3, pady=1)
        self.model_display_frame = ttk.Frame(model_frame)
        self.model_display_frame.pack(fill=tk.X, padx=6, pady=(0, 1))
        self._update_model_display()
        
        # Create batch processing frame
        self.batch_frame = ttk.LabelFrame(main_container, text="Batch Processing")
        self.batch_frame.pack(fill=tk.X, padx=3, pady=2)
        
        batch_controls = ttk.Frame(self.batch_frame)
        batch_controls.pack(fill=tk.X, padx=3, pady=1)
        
        ttk.Button(batch_controls, text="Process Directory", command=self.batch_process_directory).grid(row=0, column=0, padx=3, pady=1, sticky='w')
        
        # Parallel count control
        ttk.Label(batch_controls, text="Parallel Videos:").grid(row=0, column=1, padx=(8, 3), pady=1, sticky='w')
        self.parallel_count_var = tk.IntVar(value=2)
        parallel_spinbox = ttk.Spinbox(batch_controls, from_=1, to=4, textvariable=self.parallel_count_var, width=5)
        parallel_spinbox.grid(row=0, column=2, padx=1, pady=1, sticky='w')
        
        # Auto-save checkbox (always enabled for batch)
        self.batch_autosave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(batch_controls, text="Auto-save Results", variable=self.batch_autosave_var).grid(row=0, column=3, padx=(8, 3), pady=1, sticky='w')
        self.batch_tracking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(batch_controls, text="Tracking ID", variable=self.batch_tracking_var).grid(row=0, column=4, padx=3, pady=1, sticky='w')
        
        # Progress display
        self.batch_progress_label = ttk.Label(self.batch_frame, text="Ready for batch processing", foreground='blue')
        self.batch_progress_label.pack(fill=tk.X, padx=5, pady=(1, 2))
        
        # Progress bar
        self.batch_progress_bar = ttk.Progressbar(self.batch_frame, mode='determinate', length=300)
        self.batch_progress_bar.pack(fill=tk.X, padx=5, pady=2)
        
        # Cancel button (initially disabled)
        self.batch_cancel_btn = ttk.Button(self.batch_frame, text="Cancel Batch", command=self.cancel_batch_processing, state=tk.DISABLED)
        self.batch_cancel_btn.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Create single video / real-time inference frame (separate from batch)
        self.single_video_frame = ttk.LabelFrame(main_container, text="Single Video / Real-time Inference")
        self.single_video_frame.pack(fill=tk.X, padx=3, pady=2)
        
        single_video_controls = ttk.Frame(self.single_video_frame)
        single_video_controls.pack(fill=tk.X, padx=3, pady=1)
        
        ttk.Button(single_video_controls, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=3, pady=1)
        ttk.Button(single_video_controls, text="Open URL", command=self.open_video_url).pack(side=tk.LEFT, padx=3, pady=1)
        ttk.Button(single_video_controls, text="Close Video", command=self.close_video).pack(side=tk.LEFT, padx=3, pady=1)
        # Video and controls share the remaining window. The video column grows and shrinks with the window.
        self.content_row = ttk.Frame(main_container)
        self.content_row.pack(fill=tk.BOTH, expand=True)
        self.content_row.columnconfigure(0, weight=11)
        self.content_row.columnconfigure(1, weight=0, minsize=500)
        self.content_row.rowconfigure(0, weight=1)

        self.video_frame = ttk.Frame(self.content_row)
        self.video_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        view_bg = ttk.Style().lookup('TFrame', 'background') or '#d9d9d9'
        self.video_canvas = tk.Canvas(self.video_frame, bg=view_bg, highlightthickness=0, bd=0)
        self.video_canvas.grid(row=0, column=0, sticky='nsew')
        self.video_canvas.bind('<Configure>', self._on_video_canvas_configure)
        self._canvas_size = (960, 540)
        self._canvas_image_id = None

        # Progress/seek scale placed below the video (like a media player)
        video_progress_frame = ttk.Frame(self.video_frame)
        video_progress_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=(4, 8))
        self.progress_var = tk.IntVar(value=0)
        self.progress_scale = tk.Scale(video_progress_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                       variable=self.progress_var, showvalue=False, command=self._on_seek)
        self.progress_scale.pack(fill=tk.X, side=tk.TOP)
        self.progress_scale.bind('<Button-1>', self._start_seek)
        self.progress_scale.bind('<ButtonRelease-1>', self._end_seek)
        # Time label at right of progress bar
        self.time_label = ttk.Label(video_progress_frame, text="00:00 / 00:00")
        self.time_label.pack(anchor='e')
        
        # Keep the controls fixed; only the Detection Classes list scrolls.
        control_frame = ttk.Frame(self.content_row, width=560)
        control_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        control_frame.grid_propagate(False)
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(2, weight=1)
        
        # No need for model-specific class selection
        self.selected_classes = set()  # Will store selected unified class indices

        self.class_header_label = ttk.Label(control_frame, text="Detection Classes", font=('Arial', 12, 'bold'))
        self.class_header_label.grid(row=1, column=0, pady=(2, 3))

        # Create frame for checkboxes with scrollbar
        checkbox_container = ttk.Frame(control_frame)
        checkbox_container.grid(row=2, column=0, sticky='nsew')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(checkbox_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas for scrolling
        canvas = tk.Canvas(checkbox_container, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)

        # Create frame for checkboxes inside canvas
        checkbox_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=checkbox_frame, anchor='nw')

        self.checkboxes = {}
        self.checkbox_frame = checkbox_frame
        self.canvas = canvas
        self._update_class_checkboxes()
        
        # Add Select All/None buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=2)
        
        ttk.Button(btn_frame, text="Select All", 
                  command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select None", 
                  command=self.select_none).pack(side=tk.LEFT, padx=5)
        
        # Testing logic selector in the right-side panel
        logic_frame = ttk.Frame(control_frame)
        logic_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=(2, 2))
        ttk.Label(logic_frame, text="Test Logic", font=('Arial', 10, 'bold')).pack(anchor='w')
        logic_options = ttk.Frame(logic_frame)
        logic_options.pack(fill=tk.X, pady=0)

        self.logic_none_var = tk.BooleanVar(value=True)
        self.logic_fire_var = tk.BooleanVar(value=False)
        self.logic_sound_var = tk.BooleanVar(value=False)
        self.logic_count_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(logic_options, text="None", variable=self.logic_none_var, command=self._sync_logic_checkbox_group).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(logic_options, text="Fire Alert", variable=self.logic_fire_var, command=self._sync_logic_checkbox_group).grid(row=0, column=1, sticky='w')
        ttk.Checkbutton(logic_options, text="Sound Alert", variable=self.logic_sound_var, command=self._sync_logic_checkbox_group).grid(row=1, column=0, sticky='w')
        ttk.Checkbutton(logic_options, text="Object Counting", variable=self.logic_count_var, command=self._sync_logic_checkbox_group).grid(row=1, column=1, sticky='w')
        ttk.Button(logic_frame, text="Apply Logic", command=self.apply_selected_logic).pack(fill=tk.X, pady=(1, 0))

        self.fire_logic_mode = set()
        bottom_controls = ttk.Frame(control_frame)
        bottom_controls.grid(row=4, column=0, sticky='new')

        # Font size slider for detection labels
        font_frame = ttk.Frame(bottom_controls)
        font_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(font_frame, text="Label Font Size", font=('Arial', 10, 'bold')).pack(anchor='w')
        # Use a DoubleVar so we can have fractional scales compatible with cv2.putText fontScale
        self.font_size_var = tk.DoubleVar(value=2.0)
        # tk.Scale gives a simple horizontal slider; values from 0.2 to 3.0 are a reasonable range
        self.font_scale = tk.Scale(font_frame, from_=0.2, to=3.0, resolution=0.1,
                                   orient=tk.HORIZONTAL, variable=self.font_size_var,
                                   length=300, sliderlength=14, width=10, showvalue=True,
                                   bd=1, highlightthickness=0)
        self.font_scale.pack(fill=tk.X)

        # Confidence threshold slider (global for all classes)
        conf_frame = ttk.Frame(bottom_controls)
        conf_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(conf_frame, text="Confidence Threshold", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.conf_var = tk.DoubleVar(value=0.25)
        # showvalue False to keep compact; display numeric next to it
        self.conf_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                   variable=self.conf_var, command=self._on_conf_change,
                                   length=300, sliderlength=14, width=10,
                                   bd=1, highlightthickness=0)
        self.conf_scale.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.conf_value_label = ttk.Label(conf_frame, text=f"{self.conf_var.get():.2f}")
        self.conf_value_label.pack(side=tk.LEFT, padx=6)

        # Video control buttons (Open, Play/Pause, Forward, Backward) and speed control
        ctrl_frame = ttk.Frame(bottom_controls)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ctrl_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Open URL", command=self.open_video_url).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="⏮️ 10s", command=self.skip_backward).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(ctrl_frame, text="Pause", command=self.play_pause)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="⏭️ 10s", command=self.skip_forward).pack(side=tk.LEFT, padx=2)
        
        # Playback speed control
        speed_frame = ttk.Frame(bottom_controls)
        speed_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(speed_frame, text="Playback Speed", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = tk.Scale(speed_frame, from_=0.25, to=4.0, resolution=0.25,
                                   orient=tk.HORIZONTAL, variable=self.speed_var,
                                   command=self._on_speed_change, length=300,
                                   sliderlength=14, width=10, bd=1,
                                   highlightthickness=0)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        status_frame = ttk.Frame(bottom_controls)
        status_frame.pack(fill=tk.X, padx=5, pady=1)
        self.status_label = ttk.Label(status_frame, text="Ready - Press Space to play/pause, ←/→ for 10s skip", 
                                     font=('Arial', 9), foreground='blue')
        self.status_label.pack(anchor='w')
        self.apply_selected_logic()
        
        # Save output checkbox
        save_frame = ttk.Frame(bottom_controls)
        save_frame.pack(fill=tk.X, padx=5, pady=1)
        save_options = ttk.Frame(save_frame)
        save_options.pack(fill=tk.X)
        self.save_var = tk.BooleanVar(value=False)
        self.save_check = ttk.Checkbutton(save_options, text="Save Output Video", 
                                         variable=self.save_var, command=self._on_save_toggle)
        self.save_check.pack(side=tk.LEFT)
        self.video_tracking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_options, text="Tracking ID", variable=self.video_tracking_var).pack(side=tk.LEFT, padx=(5, 0))
        self.video_conf_display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_options, text="Show Confidence", variable=self.video_conf_display_var).pack(side=tk.LEFT, padx=(5, 0))
        # Save path label - shows where output will be saved
        self.save_path_label = ttk.Label(save_frame, text=self.output_video_path,
                                        font=('Arial', 8), foreground='gray', wraplength=300)
        self.save_path_label.pack(fill=tk.X, padx=5, pady=(2, 0))

        # Playback / capture state
        self.playing = True
        self.seeking = False
        self.was_playing_before_seek = False
        self.total_frames = 0
        self.fps = 30.0
        self.cap = None
        self.last_frame_time = 0
        self.frame_skip_count = 0
        self.max_frame_skip = 3  # Skip up to 3 frames if lagging

        # Keep model inference off Tk's event loop. Video decoding and drawing
        # can then continue at the source FPS while the worker catches up.
        self._inference_requests = Queue(maxsize=1)
        self._inference_stop = threading.Event()
        self._inference_result_lock = threading.Lock()
        self._latest_inference = None
        self._video_generation = 0
        self._inference_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._inference_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self._close_application)
        self.root.after(100, self._poll_batch_progress)

    def _inference_worker(self):
        while not self._inference_stop.is_set():
            try:
                request = self._inference_requests.get(timeout=0.1)
            except Exception:
                continue

            if request is None:
                break

            mode = request["mode"]
            frame = request["frame"]
            model_infos = request["models"]
            output = {}
            for model_info in model_infos:
                if self._inference_stop.is_set():
                    return
                model = model_info["model"]
                try:
                    task = getattr(model, "task", "")
                    if request["tracking"] and task != "semantic":
                        results = model.track(
                            frame, persist=True, tracker="botsort.yaml", verbose=False
                        )
                    else:
                        results = model(frame, verbose=False)
                    result = results[0] if isinstance(results, (list, tuple)) and results else results
                    if result is not None:
                        selected = request["selected"].get(id(model), set())
                        self._filter_result_by_names(
                            result, model, selected, request["confidence"]
                        )
                    output[id(model)] = result
                except Exception as exc:
                    print(f"Inference error ({model_info.get('name', 'model')}): {exc}")
                    output[id(model)] = None

            prediction = {
                "mode": mode,
                "signature": request["signature"],
                "active_idx": request["active_idx"],
                "video_generation": request["video_generation"],
                "by_model": output,
            }
            with self._inference_result_lock:
                self._latest_inference = prediction

    def _inference_mode_and_signature(self):
        if self.compare_var.get():
            mode = "compare"
            infos = list(self.models)
        elif self.mix_var.get() and len(self.models) >= 2:
            mode = "mix"
            infos = list(self.models[:2])
            if "counting" in self.fire_logic_mode and self.active_model_idx >= 2:
                infos.append(self.models[self.active_model_idx])
        elif self.models:
            mode = "single"
            infos = [self.models[self.active_model_idx]]
        else:
            return "none", [], (), {}

        signature = tuple(id(info["model"]) for info in infos)
        selected = {}
        for info in infos:
            model = info["model"]
            if mode == "mix":
                configured = info.get("selected_classes")
                selected[id(model)] = set(
                    model.names.values() if configured is None else configured
                )
            else:
                selected[id(model)] = {
                    self.class_names[idx]
                    for idx in self.selected_classes
                    if idx in self.class_names
                }
        return mode, infos, signature, selected

    def _submit_inference_frame(self, frame):
        mode, model_infos, signature, selected = self._inference_mode_and_signature()
        if mode == "none":
            return
        request = {
            "mode": mode,
            "models": model_infos,
            "signature": signature,
            "active_idx": self.active_model_idx,
            "video_generation": self._video_generation,
            "selected": selected,
            "confidence": float(self.conf_var.get()) if hasattr(self, "conf_var") else 0.25,
            "tracking": (
                (bool(self.video_tracking_var.get()) if hasattr(self, "video_tracking_var") else False)
                or "counting" in self.fire_logic_mode
            ),
            "frame": np.ascontiguousarray(frame.copy()),
        }
        try:
            self._inference_requests.put_nowait(request)
        except Exception:
            try:
                self._inference_requests.get_nowait()
            except Exception:
                pass
            try:
                self._inference_requests.put_nowait(request)
            except Exception:
                pass

    def _get_latest_inference(self):
        mode, _infos, signature, _selected = self._inference_mode_and_signature()
        with self._inference_result_lock:
            prediction = self._latest_inference
        if (
            prediction is None
            or prediction["mode"] != mode
            or prediction["signature"] != signature
            or prediction["active_idx"] != self.active_model_idx
            or prediction["video_generation"] != self._video_generation
        ):
            return None
        return prediction

    def _close_application(self):
        self._inference_stop.set()
        try:
            self._inference_requests.put_nowait(None)
        except Exception:
            pass
        capture = getattr(self, "cap", None)
        if capture is not None:
            capture.release()
        if self.writer is not None:
            self.writer.release()
            self.writer_size = None
        self.root.destroy()
    
    def select_all(self):
        """Select all classes in the unified list, or both model lists in mix mode"""
        if self.mix_var.get():
            for model_info in self.models[:2]:
                model_info['selected_classes'] = set(model_info['model'].names.values())
            self._update_class_checkboxes()
            return
        for idx, var in self.checkboxes.items():
            var.set(True)
            self.selected_classes.add(idx)
    
    def select_none(self):
        """Deselect all classes in the unified list, or both model lists in mix mode"""
        if self.mix_var.get():
            for model_info in self.models[:2]:
                model_info['selected_classes'] = set()
            self._update_class_checkboxes()
            return
        for idx, var in self.checkboxes.items():
            var.set(False)
            self.selected_classes.discard(idx)
    
    def update_selection(self, idx, var):
        """Update the unified selected classes list"""
        if var.get():
            self.selected_classes.add(idx)
        else:
            self.selected_classes.discard(idx)

    def _on_video_canvas_configure(self, event):
        if event.width > 50 and event.height > 50:
            self._canvas_size = (event.width, event.height)

    def _display_view_size(self):
        """Current video area in pixels, falling back until the window is mapped."""
        width = self.video_canvas.winfo_width()
        height = self.video_canvas.winfo_height()
        if width < 50 or height < 50:
            width, height = self._canvas_size
        if width < 50 or height < 50:
            width = max(640, self.root.winfo_width() - 420)
            height = max(360, self.root.winfo_height() - 260)
        return (
            max(160, min(width, MAX_PREVIEW_WIDTH)),
            max(120, min(height, MAX_PREVIEW_HEIGHT)),
        )

    def _show_rgb_frame(self, rgb_frame):
        """Scale a frame to the video area, keeping its aspect ratio, and draw it centered."""
        if Image is None or ImageTk is None:
            self.video_canvas.delete('all')
            self.video_canvas.create_text(
                20,
                20,
                anchor='nw',
                text='Video preview requires Pillow with ImageTk support.',
            )
            return
        view_w, view_h = self._display_view_size()
        frame_h, frame_w = rgb_frame.shape[:2]
        if frame_w < 1 or frame_h < 1:
            return
        scale = min(view_w / float(frame_w), view_h / float(frame_h))
        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))
        if new_w != frame_w or new_h != frame_h:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=interp)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)
        x = max(0, (view_w - new_w) // 2)
        y = max(0, (view_h - new_h) // 2)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.video_canvas.create_image(x, y, anchor='nw', image=photo)
        else:
            self.video_canvas.coords(self._canvas_image_id, x, y)
            self.video_canvas.itemconfig(self._canvas_image_id, image=photo)
        self.video_canvas.image = photo

    def _clear_video_display(self):
        self.video_canvas.delete('all')
        self.video_canvas.image = None
        self._canvas_image_id = None

    def _sync_logic_checkbox_group(self):
        """Keep the logic panel state managed by the dedicated logic helper."""
        sync_logic_checkbox_group(
            self.logic_none_var,
            self.logic_fire_var,
            self.logic_sound_var,
            self.logic_count_var,
        )

    def apply_selected_logic(self):
        """Apply the selected logic combination from the right-side testing panel."""
        self.fire_logic_mode = resolve_selected_logic(
            self.logic_none_var,
            self.logic_fire_var,
            self.logic_sound_var,
            self.logic_count_var,
        )

        if not self.fire_logic_mode:
            self.fire_alert_active = False
            self._reset_count_smoothing()
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Detection logic disabled")
            return

        if "counting" not in self.fire_logic_mode:
            self._reset_count_smoothing()

        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Applied logic: {', '.join(sorted(self.fire_logic_mode)) if self.fire_logic_mode else 'None'}")

    def _reset_count_smoothing(self):
        """Clear count smoothing history when source or mode changes."""
        self.count_history = []
        self.line_counter.reset()

    def _smooth_count_summary(self, summary):
        """Return a moving-average count summary to reduce frame-by-frame flicker."""
        snapshot = {}
        if isinstance(summary, dict):
            for label, value in summary.items():
                try:
                    snapshot[str(label)] = max(0, int(value))
                except Exception:
                    snapshot[str(label)] = 0

        self.count_history.append(snapshot)
        window = max(1, int(getattr(self, 'count_smoothing_window', 5)))
        if len(self.count_history) > window:
            self.count_history = self.count_history[-window:]

        labels = []
        for frame_counts in self.count_history:
            for label in frame_counts.keys():
                if label not in labels:
                    labels.append(label)

        if not labels:
            return {}

        smoothed = {}
        history_len = float(len(self.count_history))
        for label in labels:
            total = 0.0
            for frame_counts in self.count_history:
                total += float(frame_counts.get(label, 0))
            smoothed[label] = int(round(total / history_len))
        return smoothed

    def _extract_text_from_ocr_result(self, result):
        """Extract plate text from a YOLO OCR result by ordering characters from left to right."""
        if result is None:
            return ""

        tokens = []
        try:
            results_list = result if isinstance(result, (list, tuple)) else [result]
            for item in results_list:
                if item is None:
                    continue
                boxes = getattr(item, 'boxes', None)
                if boxes is None:
                    continue
                names = getattr(item, 'names', {}) or {}
                cls_tensor = getattr(boxes, 'cls', None)
                xyxy_tensor = getattr(boxes, 'xyxy', None)
                if cls_tensor is None or xyxy_tensor is None:
                    continue
                cls_values = cls_tensor.cpu().numpy() if hasattr(cls_tensor, 'cpu') else np.asarray(cls_tensor)
                xyxy_values = xyxy_tensor.cpu().numpy() if hasattr(xyxy_tensor, 'cpu') else np.asarray(xyxy_tensor)
                for i, cls_val in enumerate(cls_values):
                    try:
                        cls_id = int(float(cls_val))
                    except Exception:
                        continue
                    x1 = float(xyxy_values[i][0])
                    label = str(names.get(cls_id, cls_id)).strip()
                    if label and label != 'None':
                        tokens.append((x1, label))
        except Exception:
            return ""

        if not tokens:
            return ""

        tokens.sort(key=lambda item: item[0])
        text = "".join(token for _, token in tokens)
        text = text.upper().replace(" ", "")
        cleaned = []
        for ch in text:
            if ch.isalnum():
                cleaned.append(ch)
        text = "".join(cleaned)

        if len(text) < 2:
            return text
        return text

    def _read_plate_ocr(self, plate_crop):
        """Read plate text from a crop using EasyOCR, with a preprocessing fallback for noisy frames."""
        if plate_crop is None or plate_crop.size == 0:
            return ""
        if self.ocr_reader is None:
            return ""

        try:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] > 0 and gray.shape[1] > 0:
                gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                results = self.ocr_reader.readtext(
                    thresh,
                    detail=0,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                )
                if not results:
                    results = self.ocr_reader.readtext(
                        gray,
                        detail=0,
                        paragraph=False,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    )
                text = " ".join(str(item).strip() for item in results if str(item).strip())
                text = text.replace("\n", " ").replace("\r", " ")
                text = " ".join(text.split())
                return text.upper()
        except Exception:
            return ""
        return ""

    def _draw_number_plate_text(self, frame):
        """Detect plate model and OCR text, then render the read text in the top-right box."""
        if frame is None:
            return frame

        plate_model = None
        ocr_model = None
        for model_info in self.models:
            model_name = str(model_info['name']).lower()
            if 'plate' in model_name or 'license' in model_name or 'licence' in model_name:
                plate_model = model_info['model']
            if 'ocr' in model_name or 'text' in model_name:
                ocr_model = model_info['model']

        if plate_model is None:
            return frame

        try:
            result = plate_model(frame, verbose=False)
            result_obj = result[0] if isinstance(result, (list, tuple)) and result else result
            boxes = getattr(result_obj, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                self.plate_text = ""
                return frame

            best_box = None
            best_conf = -1.0
            for box in boxes:
                conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is None:
                self.plate_text = ""
                return frame

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            ocr_text = ""
            if plate_crop.size > 0:
                ocr_text = self._read_plate_ocr(plate_crop)
                if not ocr_text and ocr_model is not None:
                    try:
                        ocr_result = ocr_model(plate_crop, verbose=False)
                        ocr_text = self._extract_text_from_ocr_result(ocr_result)
                    except Exception:
                        ocr_text = ""

            # Only show an OCR panel when a real plate text has been extracted.
            if not ocr_text or len(ocr_text.strip()) < 2:
                self.plate_text = ""
                return frame

            tracking_id = None
            try:
                box_ids = getattr(best_box, 'id', None)
                if box_ids is not None:
                    if hasattr(box_ids, '__len__'):
                        tracking_id = int(box_ids[0])
                    else:
                        tracking_id = int(box_ids)
            except Exception:
                tracking_id = None

            if tracking_id is None:
                try:
                    box_ids = getattr(result_obj.boxes, 'id', None)
                    if box_ids is not None:
                        if hasattr(box_ids, '__len__'):
                            tracking_id = int(box_ids[0])
                        else:
                            tracking_id = int(box_ids)
                except Exception:
                    tracking_id = None

            self.plate_text = ocr_text.strip()
            plate_h, plate_w = frame.shape[:2]
            box_x1, box_y1 = max(0, plate_w - 420), 18
            box_x2, box_y2 = plate_w - 20, 108
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
            cv2.putText(overlay, self.plate_text[:20], (box_x1 + 12, box_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            tracker_label = f"ID {tracking_id}" if tracking_id is not None else "Number Plate"
            cv2.putText(overlay, tracker_label, (box_x1 + 12, box_y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        except Exception:
            self.plate_text = ""
            return frame

    def update(self):
        # Choose capture: prefer attached capture on GUI, fall back to module-level cap
        capture = getattr(self, 'cap', None) or globals().get('cap')
        if capture is None:
            # No video opened yet — keep the update loop alive so the UI can respond when a video is opened
            self.root.after(200, self.update)
            return

        try:
            # Read frame (only advance when playing)
            if self.playing and not self.seeking:
                # Simple frame reading - let OpenCV handle timing
                ret, frame = capture.read()
                if not ret:
                    # Live streams and some URLs have no duration; wait for the next frame
                    if getattr(self, 'is_live_source', False):
                        self.status_label.config(text="Waiting for stream...")
                        self.root.after(200, self.update)
                        return
                    # End of video - loop back to start
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = capture.read()
                    if not ret:
                        print("Error: Cannot read video frame")
                        self.root.after(1000, self.update)
                        return
            else:
                # Live URLs cannot seek backward; hold the last good frame while paused
                if getattr(self, 'is_live_source', False) and getattr(self, '_last_raw_frame', None) is not None:
                    frame = self._last_raw_frame
                else:
                    # If paused, show current frame without advancing
                    current_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    # Rewind slightly to get current frame
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
                    ret, frame = capture.read()
                    # Restore position
                    capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    if not ret:
                        print("Error: Cannot read paused frame")
                        self.root.after(1000, self.update)
                        return

        except Exception as e:
            print(f"Error reading video frame: {str(e)}")
            # Try to recover the video stream on next update
            self.root.after(33, self.update)
            return

        if getattr(self, 'is_live_source', False):
            self._last_raw_frame = frame.copy()
        
        # Check if we have any models loaded
        if not self.models:
            # Display frame without inference
            self._show_rgb_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.root.after(33, self.update)
            return

        self._submit_inference_frame(frame)
        prediction = self._get_latest_inference()

        # Render the newest completed prediction over the current video frame.
        count_summary = {}
        if self.compare_var.get():
            all_results = [
                prediction["by_model"].get(id(info["model"])) if prediction else None
                for info in self.models
            ]
            
            # Create a grid of results (2x2 format)
            h, w = frame.shape[:2]
            n_models = len(self.models)
            
            # Use 2x2 grid view for comparison
            combined_w = w * 2
            combined_h = h * 2
            combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            
            # Process each model's results
            for i, (results, model_info) in enumerate(zip(all_results, self.models)):
                if i >= 4: break # Max 4 models in 2x2 grid
                
                # Make a fresh copy of the frame for this model
                model_frame = frame.copy()
                
                has_boxes = getattr(results, 'boxes', None) is not None
                task = getattr(model_info['model'], 'task', '')
                is_semantic = (task == 'semantic')
                # Keep count summary in sync with the active model panel in compare mode.
                if i == self.active_model_idx and results is not None:
                    count_summary = summarize_model_counts(results, model_info['model'].names)
                
                try:
                    # First plot the base frame with detections
                    annotated_frame = model_frame.copy()
                    
                    # Plot using YOLO's visualization
                    if results is not None:
                        plot_masks = getattr(results, 'masks', None) is not None or getattr(results, 'semantic_mask', None) is not None
                        plot_boxes = has_boxes and not is_semantic and not plot_masks
                        annotated_frame = results.plot(boxes=plot_boxes, masks=plot_masks)
                        
                        # Manual enhancement of visualization
                        if plot_boxes and hasattr(results, 'boxes') and results.boxes is not None:
                            boxes = results.boxes
                            for box in boxes:
                                # Get box coordinates
                                if hasattr(box, 'xyxy'):
                                    try:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        # Draw thicker box
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                        
                                        # Add label with confidence
                                        if hasattr(box, 'cls') and hasattr(box, 'conf'):
                                            cls_id = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            label = f"{results.names[cls_id]} {conf:.2f}"
                                            
                                            # Draw label background
                                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 4),
                                                        (x1 + text_size[0], y1), (0, 255, 0), -1)
                                            # Draw text
                                            cv2.putText(annotated_frame, label, (x1, y1 - 2),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                    except Exception as e:
                                        print(f"Error drawing box: {str(e)}")
                                        continue

                    if results is not None:
                        annotated_frame = self._merge_result_overlay(
                            model_frame, annotated_frame, results
                        )
                
                except Exception as e:
                    print(f"Error in detection visualization: {str(e)}")
                    annotated_frame = model_frame.copy()
                
                try:
                    # Draw model info and detection count
                    num_dets = len(results.boxes) if hasattr(results, 'boxes') and results.boxes is not None else 0
                    info_text = f"Model: {model_info['name']} ({num_dets} detections)"
                    # Add black background for better text visibility
                    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_frame, (8, 8), (text_size[0] + 12, 40), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, info_text, 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add to combined frame (2x2 grid)
                    x_offset = (i % 2) * w
                    y_offset = (i // 2) * h
                    combined_frame[y_offset:y_offset+h, x_offset:x_offset+w] = annotated_frame
                except Exception as e:
                    print(f"Error adding frame to grid: {str(e)}")
            
            # Use the combined frame for display
            annotated_frame = combined_frame
        elif self.mix_var.get() and len(self.models) >= 2:
            cached_results = prediction["by_model"] if prediction else {}
            annotated_frame = self._render_mix(frame, cached_results)
            if "counting" in self.fire_logic_mode and self.models:
                try:
                    active_info = self.models[self.active_model_idx]
                    active_model = active_info['model']
                    active_result = cached_results.get(id(active_model))
                    if active_result is not None:
                        selected = active_info.get('selected_classes')
                        if selected is None:
                            selected = set(active_model.names.values())
                        conf_thresh = float(self.conf_var.get()) if hasattr(self, 'conf_var') else 0.25
                        self._filter_result_by_names(active_result, active_model, selected, conf_thresh)
                        count_summary = summarize_model_counts(active_result, active_model.names)
                except Exception:
                    pass
        else:
            # Single model mode - use active model
            model_info = self.models[self.active_model_idx]
            task = getattr(model_info['model'], 'task', '')
            is_semantic = (task == 'semantic')
            raw_result = prediction["by_model"].get(id(model_info['model'])) if prediction else None
            results = [raw_result] if raw_result is not None else None

            if raw_result is None:
                annotated_frame = frame.copy()
            else:
                has_boxes = getattr(raw_result, 'boxes', None) is not None
                count_summary = summarize_model_counts(raw_result, model_info['model'].names)
                plot_masks = getattr(raw_result, 'masks', None) is not None or getattr(raw_result, 'semantic_mask', None) is not None
                plot_boxes = has_boxes and not is_semantic and not plot_masks
                annotated_frame = raw_result.plot(
                    boxes=plot_boxes,
                    masks=plot_masks,
                    labels=False,
                    conf=False,
                    line_width=2
                )
                self._apply_custom_mask_colors(annotated_frame, raw_result)
                try:
                    self._draw_labels(annotated_frame, raw_result)
                except Exception:
                    pass
                annotated_frame = self._merge_result_overlay(
                    frame, annotated_frame, raw_result
                )
        # mask calculation moved inside comparison loop for each model

        active_model = None
        if self.models:
            active_model = self.models[self.active_model_idx]['model']

        fire_detected = False
        fire_logic_requested = bool(
            {"fire_alert", "sound_alert"} & self.fire_logic_mode
        )
        if fire_logic_requested:
            fire_detected, _ = self.fire_detector.detect(frame, model=active_model)
            self.fire_alert_active = fire_detected

            if fire_detected and "sound_alert" in self.fire_logic_mode:
                now = time.time()
                if now - self.last_fire_alert_time >= 1.5:
                    self.fire_detector.play_alert_sound()
                    self.last_fire_alert_time = now
        else:
            self.fire_alert_active = False

        if "counting" in self.fire_logic_mode and active_model is not None:
            try:
                active_result = None
                if prediction:
                    active_result = prediction["by_model"].get(id(active_model))
                if active_result is not None:
                    count_summary, line_y = self.line_counter.update(
                        active_result, active_model.names
                    )
                    annotated_frame = draw_counting_line(annotated_frame, line_y)
                else:
                    count_summary = dict(self.line_counter.counts)
                annotated_frame = draw_count_summary_overlay(
                    annotated_frame,
                    count_summary,
                    title=f"{self.models[self.active_model_idx]['name']} Counts",
                )
            except Exception:
                pass

        annotated_frame = self._draw_number_plate_text(annotated_frame)

        # Fit the annotated frame into the current video area
        self._show_rgb_frame(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        # Keep the output dimensions stable when comparison mode creates a grid.
        if self.save_var.get() and self.writer is not None:
            output_frame = annotated_frame
            if self.writer_size and output_frame.shape[1::-1] != self.writer_size:
                output_frame = cv2.resize(output_frame, self.writer_size)
            self.writer.write(output_frame)

        # Update progress UI
        try:
            pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if getattr(self, 'is_live_source', False):
                self.time_label.config(text="LIVE")
            elif self.total_frames:
                self.progress_var.set(min(pos, max(0, self.total_frames - 1)))
                # update time label
                cur_sec = pos / max(1.0, self.fps)
                tot_sec = self.total_frames / max(1.0, self.fps)
                self.time_label.config(text=f"{self._format_time(cur_sec)} / {self._format_time(tot_sec)}")
        except Exception:
            pass
        
        # Calculate next update time based on speed and actual frame rate
        # Use actual FPS from video for smooth playback
        base_interval = int(1000 / max(1.0, self.fps))  # milliseconds per frame
        # Adjust by speed factor
        next_update = int(base_interval / self.speed_var.get())

        # Check if we're lagging and need to skip frames
        current_time = self.root.winfo_exists() and self.root.tk.call('clock', 'milliseconds') or 0
        if self.last_frame_time > 0:
            actual_interval = current_time - self.last_frame_time
            if actual_interval > next_update * 2:  # If we're taking more than 2x expected time
                self.frame_skip_count += 1
                if self.frame_skip_count <= self.max_frame_skip:
                    # Skip next frame by reducing update interval
                    next_update = max(16, next_update // 2)
                else:
                    # Reset skip count after max skips
                    self.frame_skip_count = 0
            else:
                self.frame_skip_count = 0

        self.last_frame_time = current_time

        # Cap minimum interval to prevent overwhelming the UI
        next_update = max(16, next_update)  # Minimum 16ms (~60fps)
        # Schedule the next update
        self.root.after(next_update, self.update)

    def attach_capture(self, capture):
        """Attach a cv2.VideoCapture to the GUI and configure progress/output."""
        global out, cap
        previous_capture = getattr(self, "cap", None)
        if previous_capture is not None and previous_capture is not capture:
            previous_capture.release()
        cap = capture
        self.cap = capture
        self._video_generation += 1
        with self._inference_result_lock:
            self._latest_inference = None
        while True:
            try:
                self._inference_requests.get_nowait()
            except Exception:
                break
        if capture is None or not capture.isOpened():
            self.total_frames = 0
            self.fps = 30.0
            self.progress_scale.config(from_=0, to=0)
            return

        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 1:
            self.fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Streams and some URLs do not report a duration
        self.is_live_source = self.total_frames <= 1
        if self.is_live_source:
            self.progress_scale.config(state='disabled', from_=0, to=0)
            self.progress_var.set(0)
            self.time_label.config(text="LIVE")
        else:
            self.progress_scale.config(state='normal', from_=0, to=max(0, self.total_frames - 1))
            self.progress_var.set(0)

        # Close existing writer if present
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.writer_size = None
        
        # Create new writer if saving is enabled
        if self.save_var.get():
            output_path = self.output_video_path
            if not output_path:
                output_path = self._prompt_for_output_path()
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                if not self.writer.isOpened():
                    self.writer = None
                    self.save_var.set(False)
                    messagebox.showerror("Error", f"Could not create output file: {output_path}")
                else:
                    self.writer_size = (width, height)

    def open_video(self):
        path = filedialog.askopenfilename(title="Select video file",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not path:
            return
        self._open_video_source(path, from_url=False)

    def open_video_url(self):
        """Open a video from a direct URL (http, https, or rtsp)."""
        url = self.url_var.get().strip() if hasattr(self, 'url_var') else ''
        if not url:
            url = simpledialog.askstring(
                "Open Video URL",
                "Enter a direct video URL (http, https, or rtsp):",
                parent=self.root,
            )
            if not url:
                return
            url = url.strip()
            self.url_var.set(url)
        if not url:
            return
        self._open_video_source(url, from_url=True)

    def _open_video_source(self, source, from_url=False):
        capture = None
        try:
            capture = cv2.VideoCapture()
            if from_url:
                open_timeout = getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC', None)
                read_timeout = getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC', None)
                if open_timeout is not None:
                    capture.set(open_timeout, 10000)
                if read_timeout is not None:
                    capture.set(read_timeout, 10000)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            opened = capture.open(source)
            if not opened or not capture.isOpened():
                messagebox.showerror("Error", f"Could not open video:\n{source}")
                capture.release()
                return

            if not from_url:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            ret, frame = False, None
            attempts = 5 if from_url else 3
            for _ in range(attempts):
                ret, frame = capture.read()
                if ret and frame is not None and frame.size > 0:
                    break
                if not from_url:
                    capture.release()
                    capture = cv2.VideoCapture(source)

            if not ret or frame is None or frame.size == 0:
                messagebox.showerror("Error", "Could not read valid video frames from the source.")
                capture.release()
                return

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.current_video_path = source
            self.last_valid_frame = 0
            self._reset_count_smoothing()

            self.attach_capture(capture)
            self.playing = True
            try:
                self.play_button.config(text="Pause")
            except Exception:
                pass

            self.batch_frame.pack_forget()

            if self.is_live_source:
                self.status_label.config(text="Stream opened — seeking unavailable. Space: play/pause")
            else:
                self.status_label.config(text="Video loaded - Space: play/pause, ←/→: 10s skip")

        except Exception as e:
            messagebox.showerror("Error", f"Error opening video: {str(e)}")
            if capture is not None:
                capture.release()

    def close_video(self):
        """Close the current video and show batch processing section again"""
        global cap, out
        # Stop playback
        self.playing = False
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = None
        
        # Release writer if active
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.writer_size = None
        out = None
        
        # Clear video display
        self._last_raw_frame = None
        self._reset_count_smoothing()
        self._clear_video_display()
        
        # Reset progress
        self.is_live_source = False
        self.current_video_path = None
        self._video_generation += 1
        with self._inference_result_lock:
            self._latest_inference = None
        self.progress_var.set(0)
        self.progress_scale.config(state='normal', to=0)
        self.time_label.config(text="00:00 / 00:00")
        
        # Show batch processing section again
        self.batch_frame.pack_forget()
        self.single_video_frame.pack_forget()
        self.batch_frame.pack(fill=tk.X, padx=5, pady=5, before=self.content_row)
        self.single_video_frame.pack(fill=tk.X, padx=5, pady=5, before=self.content_row)
        
        # Update status
        self.status_label.config(text="Video closed - Ready")
        try:
            self.play_button.config(text="Play")
        except Exception:
            pass

    def play_pause(self):
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        status = "Playing" if self.playing else "Paused"
        self.status_label.config(text=f"{status} - Space: play/pause, ←/→: 10s skip")
        
    def skip_forward(self):
        """Skip forward 10 seconds"""
        if getattr(self, 'is_live_source', False):
            self.status_label.config(text="Seeking is unavailable for this stream")
            return
        capture = getattr(self, 'cap', None) or globals().get('cap')
        if capture is None or not capture.isOpened():
            return
        current_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
        skip_frames = int(10 * self.fps)  # 10 seconds worth of frames
        new_pos = min(current_pos + skip_frames, self.total_frames - 1)
        self.seek_to_frame(new_pos)
        self.status_label.config(text="Skipped +10s - Space: play/pause, ←/→: 10s skip")
        
    def skip_backward(self):
        """Skip backward 10 seconds"""
        if getattr(self, 'is_live_source', False):
            self.status_label.config(text="Seeking is unavailable for this stream")
            return
        capture = getattr(self, 'cap', None) or globals().get('cap')
        if capture is None or not capture.isOpened():
            return
        current_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
        skip_frames = int(10 * self.fps)  # 10 seconds worth of frames
        new_pos = max(current_pos - skip_frames, 0)
        self.seek_to_frame(new_pos)
        self.status_label.config(text="Skipped -10s - Space: play/pause, ←/→: 10s skip")
        
    def seek_to_frame(self, frame_pos):
        """Seek to a specific frame position"""
        capture = getattr(self, 'cap', None) or globals().get('cap')
        if capture is None or not capture.isOpened():
            return
        # Pause during seek
        was_playing = self.playing
        self.playing = False
        # Seek to frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.progress_var.set(frame_pos)
        # Resume if was playing
        if was_playing:
            self.playing = True
        """Handle playback speed change"""
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f}x")
        
    def _prompt_for_output_path(self):
        """Ask the user for the output video filename and return a valid path."""
        default_name = os.path.basename(self.output_video_path)
        default_dir = os.path.dirname(self.output_video_path) or os.getcwd()
        output_path = filedialog.asksaveasfilename(
            title="Save output video as",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension='.mp4',
            filetypes=[('MP4 Video', '*.mp4'), ('AVI Video', '*.avi'), ('All files', '*.*')]
        )
        if not output_path:
            return None

        if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
            output_path += '.mp4'

        self.output_video_path = output_path
        self.save_path_label.config(text=output_path)
        return output_path

    def _on_save_toggle(self):
        """Handle save checkbox toggle"""
        if self.cap is None:
            # No video loaded, uncheck the box
            self.save_var.set(False)
            messagebox.showwarning("Warning", "Please open a video first")
            return

        if self.save_var.get():
            output_path = self._prompt_for_output_path()
            if output_path is None:
                self.save_var.set(False)
                return

            # Create writer when saving is enabled
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                messagebox.showerror("Error", f"Could not create output file: {output_path}")
                self.writer = None
                self.save_var.set(False)
            else:
                self.writer_size = (width, height)
        else:
            # Close writer when saving is disabled
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.writer_size = None

    def _start_seek(self, event):
        # User started interacting with the progress slider
        self.seeking = True
        self.was_playing_before_seek = self.playing
        self.playing = False

    def _on_seek(self, value):
        # value is a string; only seek while user is dragging (self.seeking True)
        if not self.seeking:
            return
        try:
            idx = int(float(value))
        except Exception:
            return
        capture = getattr(self, 'cap', None) or globals().get('cap')
        if capture is None or not capture.isOpened():
            return
        # Seeking should stay responsive; queue inference instead of running
        # the model repeatedly while the user drags the timeline.
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if not ret:
            return
        self._submit_inference_frame(frame)
        prediction = self._get_latest_inference()
        if self.mix_var.get() and len(self.models) >= 2:
            results_by_model = prediction["by_model"] if prediction else {}
            annotated_frame = self._render_mix(frame, results_by_model)
        else:
            model = self.models[self.active_model_idx]['model'] if self.models else None
            result = (
                prediction["by_model"].get(id(model))
                if model is not None and prediction else None
            )
            if result is None:
                annotated_frame = frame
            else:
                task = getattr(model, 'task', '')
                is_semantic = task == 'semantic'
                has_boxes = getattr(result, 'boxes', None) is not None
                plot_masks = getattr(result, 'masks', None) is not None or getattr(result, 'semantic_mask', None) is not None
                annotated_frame = result.plot(
                    boxes=has_boxes and not is_semantic and not plot_masks,
                    masks=plot_masks,
                    labels=False,
                    conf=False,
                    line_width=2,
                )
                self._apply_custom_mask_colors(annotated_frame, result)
                self._draw_labels(annotated_frame, result)
        self._show_rgb_frame(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    def _end_seek(self, event):
        # Finish seeking; optionally resume playback
        self.seeking = False
        if self.was_playing_before_seek:
            self.playing = True

    def _on_conf_change(self, value):
        """Update the numeric label when confidence slider changes."""
        try:
            v = float(value)
        except Exception:
            v = float(self.conf_var.get()) if hasattr(self, 'conf_var') else 0.25
        try:
            self.conf_value_label.config(text=f"{v:.2f}")
        except Exception:
            pass

    def _on_speed_change(self, value):
        """Update playback speed label when speed slider changes."""
        try:
            v = float(value)
        except Exception:
            v = float(self.speed_var.get()) if hasattr(self, 'speed_var') else 1.0
        try:
            # Update UI label
            self.speed_label.config(text=f"{v:.2f}x")
        except Exception:
            pass

    def _format_time(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def get_selected_classes(self):
        return self.selected_classes
    
    def add_model(self):
        """Open file dialog to select a YOLO model file"""
        if YOLO is None:
            messagebox.showerror(
                "Ultralytics unavailable",
                "Install the model dependency with: pip install ultralytics",
            )
            return
        path = filedialog.askopenfilename(
            title="Select YOLO model file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if not path:
            return
            
        try:
            model = YOLO(path)
            self.models.append({
                'model': model,
                'path': path,
                'name': path.split('/')[-1],
                'selected_classes': set(model.names.values()),
            })
            
            # Update unified classes and automatically select them
            for name in model.names.values():
                if name not in self.unified_class_names:
                    idx = len(self.unified_class_names)
                    self.unified_class_names[name] = idx
                    self.class_names[idx] = name
                    self.selected_classes.add(idx)  # Auto-select new classes
            
            self._update_model_display()
            self._update_class_checkboxes()
            messagebox.showinfo("Success", f"Model loaded: {path.split('/')[-1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def remove_model(self):
        """Remove the currently active model"""
        if not self.models:
            messagebox.showwarning("Warning", "No models loaded")
            return
        self._delete_model(self.active_model_idx)
    
    def _set_active_model(self, idx):
        """Select the active model used for single-model inference and OCR."""
        if 0 <= idx < len(self.models):
            self.active_model_idx = idx
            self._update_model_display()
            self.status_label.config(text=f"Active model: {self.models[idx]['name']}")

    def _update_model_display(self):
        """Update the model display area with model names and active/delete controls."""
        for widget in self.model_display_frame.winfo_children():
            widget.destroy()
        if not self.models:
            ttk.Label(self.model_display_frame, text="No models loaded").pack(side=tk.LEFT)
        else:
            for i, model_info in enumerate(self.models):
                frame = ttk.Frame(self.model_display_frame)
                frame.pack(side=tk.LEFT, padx=2)
                if i == self.active_model_idx:
                    ttk.Label(frame, text=f"{model_info['name']} (Active)", foreground='green', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
                else:
                    ttk.Button(frame, text=model_info['name'], command=lambda idx=i: self._set_active_model(idx)).pack(side=tk.LEFT)
                ttk.Button(frame, text="Delete", command=lambda idx=i: self._delete_model(idx)).pack(side=tk.LEFT)
        # Disable add button if 4 models loaded
        for widget in self.model_buttons_frame.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == "Add Model":
                widget['state'] = tk.NORMAL if len(self.models) < 4 else tk.DISABLED

    def _delete_model(self, idx):
        del self.models[idx]
        if self.active_model_idx >= len(self.models):
            self.active_model_idx = max(0, len(self.models) - 1)
        if self.mix_var.get() and len(self.models) < 2:
            self.mix_var.set(False)
        self._update_model_display()
        self._update_class_checkboxes()
    
    def toggle_comparison(self):
        """Toggle between single model and comparison mode"""
        if self.batch_processing:
            self.compare_var.set(False)
            messagebox.showwarning("Warning", "Comparison is not available during batch processing")
            return

        if self.compare_var.get() and len(self.models) < 2:
            messagebox.showwarning("Warning", "Need at least 2 models for comparison")
            self.compare_var.set(False)
            return
        if self.compare_var.get():
            self.mix_var.set(False)
        self._update_class_checkboxes()

    def toggle_mix(self):
        """Overlay two models on one video, with a class list for each model."""
        if self.batch_processing:
            self.mix_var.set(False)
            messagebox.showwarning("Warning", "Mix mode is not available during batch processing")
            return

        if self.mix_var.get() and len(self.models) < 2:
            messagebox.showwarning("Warning", "Need at least 2 models to mix detections")
            self.mix_var.set(False)
            return

        if self.mix_var.get():
            self.compare_var.set(False)
            for model_info in self.models[:2]:
                if model_info.get('selected_classes') is None:
                    model_info['selected_classes'] = set(model_info['model'].names.values())
            self.status_label.config(text="Mix mode — M1 green, M2 orange. Pick classes for each model.")
        self._update_class_checkboxes()

    def _on_mix_class_toggle(self, model_idx, class_name, var):
        if model_idx >= len(self.models):
            return
        selected = self.models[model_idx].setdefault('selected_classes', set())
        if var.get():
            selected.add(class_name)
        else:
            selected.discard(class_name)

    def _set_model_classes(self, model_idx, enabled):
        if model_idx >= len(self.models):
            return
        model_info = self.models[model_idx]
        if enabled:
            model_info['selected_classes'] = set(model_info['model'].names.values())
        else:
            model_info['selected_classes'] = set()
        self._update_class_checkboxes()
    
    def _update_class_checkboxes(self):
        """Recreate class checkboxes. Mix mode shows one list per model."""
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        
        self.checkboxes = {}
        mixing = self.mix_var.get() and len(self.models) >= 2
        if mixing:
            self.class_header_label.config(text="Mix Classes")
            if len(self.models) > 2:
                ttk.Label(
                    self.checkbox_frame,
                    text="Using the first two loaded models",
                    font=('Arial', 8),
                    foreground='gray',
                ).pack(anchor='w', padx=5, pady=(0, 4))
            for model_index, model_info in enumerate(self.models[:2]):
                self._build_mix_class_section(model_index, model_info)
        else:
            self.class_header_label.config(text="Detection Classes")
            for idx, name in self.class_names.items():
                var = tk.BooleanVar(master=self.root, value=idx in self.selected_classes)
                cb = ttk.Checkbutton(self.checkbox_frame, text=name, variable=var,
                                   command=lambda idx=idx, var=var: self.update_selection(idx, var))
                cb.pack(anchor='w', padx=5, pady=2)
                self.checkboxes[idx] = var
        
        self.checkbox_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _build_mix_class_section(self, model_index, model_info):
        """One scrollable class list for a single model in mix mode."""
        section = ttk.Frame(self.checkbox_frame)
        section.pack(fill=tk.X, padx=4, pady=(6, 2))

        header = ttk.Frame(section)
        header.pack(fill=tk.X)
        swatch = tk.Label(header, text="    ", bg=self.mix_colors_hex[model_index], relief='solid', borderwidth=1)
        swatch.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            header,
            text=f"M{model_index + 1}: {model_info['name']}",
            font=('Arial', 9, 'bold'),
        ).pack(side=tk.LEFT)

        btn_row = ttk.Frame(section)
        btn_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Button(
            btn_row, text="All", width=6,
            command=lambda idx=model_index: self._set_model_classes(idx, True),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            btn_row, text="None", width=6,
            command=lambda idx=model_index: self._set_model_classes(idx, False),
        ).pack(side=tk.LEFT)

        selected = model_info.setdefault('selected_classes', set(model_info['model'].names.values()))
        names = model_info['model'].names
        ordered = [names[k] for k in sorted(names.keys())]
        for name in ordered:
            var = tk.BooleanVar(master=self.root, value=name in selected)
            cb = ttk.Checkbutton(
                section,
                text=name,
                variable=var,
                command=lambda n=name, v=var, idx=model_index: self._on_mix_class_toggle(idx, n, v),
            )
            cb.pack(anchor='w', padx=8, pady=1)

        ttk.Separator(self.checkbox_frame, orient='horizontal').pack(fill=tk.X, padx=4, pady=(6, 0))

    def _filter_result_by_names(self, result, model, selected_names, conf_thresh):
        """Keep detections whose class name is selected for this model."""
        if result is None or getattr(result, 'boxes', None) is None:
            return
        cls_t = getattr(result.boxes, 'cls', None)
        conf_t = getattr(result.boxes, 'conf', None)
        if cls_t is None:
            return
        names = getattr(result, 'names', None) or model.names
        mask_list = []
        for i, cls in enumerate(cls_t):
            try:
                cls_id = int(cls.item()) if hasattr(cls, 'item') else int(cls)
                name = names.get(cls_id, str(cls_id)) if hasattr(names, 'get') else str(cls_id)
                if conf_t is None:
                    conf_val = 1.0
                elif hasattr(conf_t[i], 'item'):
                    conf_val = float(conf_t[i].item())
                else:
                    conf_val = float(conf_t[i])
                mask_list.append(name in selected_names and conf_val >= conf_thresh)
            except Exception:
                mask_list.append(False)
        if not mask_list:
            return
        if torch is None:
            return
        mask = torch.tensor(mask_list, dtype=torch.bool)
        num_boxes = len(result.boxes)
        if num_boxes > 0 and len(mask) == num_boxes:
            if hasattr(result, 'masks') and result.masks is not None:
                result.masks = result.masks[mask]
            result.boxes = result.boxes[mask]

    def _overlay_instance_masks(self, frame, result, color, alpha=0.45):
        """Paint instance masks onto the shared frame. Returns True if any mask was drawn."""
        if not hasattr(result, 'masks') or result.masks is None:
            return False
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return False
        masks_data = result.masks.data
        masks_np = masks_data.cpu().numpy() if hasattr(masks_data, 'cpu') else np.array(masks_data)
        cls_ids = getattr(result.boxes, 'cls', None)
        if cls_ids is None:
            return False
        cls_arr = cls_ids.cpu().numpy() if hasattr(cls_ids, 'cpu') else np.array(cls_ids)
        names = getattr(result, 'names', {}) or {}
        h, w = frame.shape[:2]
        drew = False
        for i, mask in enumerate(masks_np):
            cls_id = int(cls_arr[i]) if i < len(cls_arr) else -1
            name = names.get(cls_id, str(cls_id))
            use_color = self.custom_class_colors.get(name, color)
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = mask > 0.5
            if not np.any(binary):
                continue
            overlay = frame.copy()
            overlay[binary] = use_color
            blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            frame[binary] = blended[binary]
            drew = True
        return drew

    def _draw_mix_legend(self, frame, models, colors):
        y = 8
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        for i, (model_info, color) in enumerate(zip(models, colors)):
            text = f"M{i + 1}: {model_info['name']}"
            (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
            row_h = th + base + 10
            cv2.rectangle(frame, (8, y), (46 + tw, y + row_h), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (14, y + 4), (32, y + row_h - 4), color, cv2.FILLED)
            cv2.putText(frame, text, (38, y + th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += row_h + 4

    def _merge_result_overlay(self, current_frame, rendered_frame, result):
        """Carry annotations from an older inference frame onto live playback."""
        source_frame = getattr(result, "orig_img", None)
        if (
            source_frame is None
            or source_frame.shape != rendered_frame.shape
            or current_frame.shape != rendered_frame.shape
        ):
            return rendered_frame
        changed = np.any(cv2.absdiff(rendered_frame, source_frame) > 12, axis=2)
        merged = current_frame.copy()
        merged[changed] = rendered_frame[changed]
        return merged

    def _render_mix(self, frame, inference_results=None):
        """Draw cached results from the first two models on the current frame."""
        annotated = frame.copy()
        conf_thresh = float(self.conf_var.get()) if hasattr(self, 'conf_var') else 0.25
        models = self.models[:2]
        colors = self.mix_colors_bgr
        for i, model_info in enumerate(models):
            model = model_info['model']
            color = colors[i % len(colors)]
            try:
                task = getattr(model, 'task', '')
                is_semantic = (task == 'semantic')
                if inference_results is None:
                    # Seeking uses this path for a one-off preview.
                    frame_copy = np.ascontiguousarray(frame)
                    use_track = (
                        hasattr(self, 'video_tracking_var')
                        and self.video_tracking_var.get()
                        and not is_semantic
                    )
                    if use_track:
                        results = model.track(frame_copy, persist=True, tracker="botsort.yaml", verbose=False)
                    else:
                        results = model(frame_copy, verbose=False)
                    result = results[0] if isinstance(results, (list, tuple)) else results
                else:
                    result = inference_results.get(id(model))
                if result is None:
                    continue
                selected = model_info.get('selected_classes')
                if selected is None:
                    selected = set(model.names.values())
                self._filter_result_by_names(result, model, selected, conf_thresh)
                drew_masks = self._overlay_instance_masks(annotated, result, color)
                if is_semantic and not drew_masks:
                    plotted = result.plot(labels=False, conf=False, boxes=False)
                    if plotted is not None and plotted.shape == annotated.shape:
                        semantic_overlay = self._merge_result_overlay(
                            annotated, plotted, result
                        )
                        annotated[:] = semantic_overlay
                self._draw_labels(
                    annotated,
                    result,
                    label_prefix=f"M{i + 1}",
                    box_color=color,
                    draw_box=not drew_masks and not is_semantic,
                )
            except Exception as e:
                print(f"Mix inference error ({model_info['name']}): {e}")
        self._draw_mix_legend(annotated, models, colors)
        return annotated

    def _apply_custom_mask_colors(self, frame, result, alpha=0.5):
        """Override mask colors for classes defined in self.custom_class_colors.
        
        frame: numpy BGR image (modified in-place)
        result: single Results object from YOLO model
        alpha: transparency for the mask overlay (0.0 - 1.0)
        """
        if not self.custom_class_colors:
            return
        if not hasattr(result, 'masks') or result.masks is None:
            return
        if not hasattr(result, 'boxes') or result.boxes is None:
            return
        
        cls_ids = getattr(result.boxes, 'cls', None)
        if cls_ids is None:
            return
        if hasattr(cls_ids, 'cpu'):
            cls_arr = cls_ids.cpu().numpy()
        else:
            cls_arr = np.array(cls_ids)
        
        masks_data = result.masks.data  # (N, H, W) tensor
        if hasattr(masks_data, 'cpu'):
            masks_np = masks_data.cpu().numpy()
        else:
            masks_np = np.array(masks_data)
        
        h, w = frame.shape[:2]
        
        for i, cls_id in enumerate(cls_arr):
            cls_id = int(cls_id)
            name = result.names.get(cls_id, str(cls_id))
            if name not in self.custom_class_colors:
                continue
            
            color_bgr = self.custom_class_colors[name]
            
            # Resize mask to frame size if needed
            mask = masks_np[i]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Create colored overlay
            overlay = frame.copy()
            overlay[binary_mask == 1] = color_bgr
            
            # Blend with original frame
            frame[binary_mask == 1] = cv2.addWeighted(
                frame, 1 - alpha, overlay, alpha, 0
            )[binary_mask == 1]

    def _draw_labels(self, frame, res, label_prefix=None, box_color=None, draw_box=False):
        """Draw labels on the provided BGR frame using current font-size slider.

        frame: numpy BGR image (modified in-place)
        res: single Results object (results[0]) from aiunika model
        label_prefix: optional tag such as M1 so mixed models stay distinct
        box_color: optional BGR color for the label chip and box
        draw_box: draw a rectangle when this result was not already drawn as masks
        """
        # Try to robustly extract boxes, classes and confidences
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return

        # xyxy
        xyxy = getattr(res.boxes, 'xyxy', None)
        if xyxy is None:
            return

        if hasattr(xyxy, 'cpu'):
            boxes_arr = xyxy.cpu().numpy()
        else:
            boxes_arr = np.array(xyxy)

        cls_ids = getattr(res.boxes, 'cls', None)
        if cls_ids is None:
            return
        if hasattr(cls_ids, 'cpu'):
            cls_arr = cls_ids.cpu().numpy()
        else:
            cls_arr = np.array(cls_ids)

        confs = getattr(res.boxes, 'conf', None)
        if confs is not None:
            if hasattr(confs, 'cpu'):
                conf_arr = confs.cpu().numpy()
            else:
                conf_arr = np.array(confs)
        else:
            conf_arr = [None] * len(cls_arr)

        ids = getattr(res.boxes, 'id', None)
        if ids is not None:
            if hasattr(ids, 'cpu'):
                id_arr = ids.cpu().numpy()
            else:
                id_arr = np.array(ids)
        else:
            id_arr = [None] * len(cls_arr)

        font_scale = float(self.font_size_var.get()) if hasattr(self, 'font_size_var') else 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, box in enumerate(boxes_arr):
            try:
                x1, y1, x2, y2 = map(int, box[:4])
            except Exception:
                continue
            cls_id = int(cls_arr[i])
            name = res.names.get(cls_id, str(cls_id))
            conf = conf_arr[i] if conf_arr is not None else None
            show_conf = hasattr(self, 'video_conf_display_var') and self.video_conf_display_var.get()
            label = f"{name} {conf:.2f}" if (show_conf and conf is not None and not np.isnan(conf)) else name
            show_tracking_id = (
                (hasattr(self, 'video_tracking_var') and self.video_tracking_var.get())
                or 'counting' in getattr(self, 'fire_logic_mode', set())
            )
            if show_tracking_id:
                tid = None
                if id_arr is not None and len(id_arr) > i:
                    try:
                        tid = int(id_arr[i])
                    except Exception:
                        tid = None
                if tid is not None and tid != -1:
                    label = f"ID {tid} | {label}"
            if label_prefix:
                label = f"{label_prefix} | {label}"

            if draw_box and box_color is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Text size and position
            thickness = max(1, int(round(font_scale * 2)))
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x1
            text_y = y1 - 6
            if text_y - text_h - baseline < 0:
                text_y = y1 + text_h + 6

            # Background rectangle for readability
            rect_tl = (text_x, text_y - text_h - baseline)
            rect_br = (text_x + text_w, text_y + baseline)
            bg = box_color if box_color is not None else (0, 0, 0)
            fg = (0, 0, 0) if box_color is not None else (255, 255, 255)
            cv2.rectangle(frame, rect_tl, rect_br, bg, cv2.FILLED)

            # Draw text (black outline then foreground for readability)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=thickness + 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, fg, thickness=thickness, lineType=cv2.LINE_AA)
    
    def _draw_batch_labels(self, frame, res):
        """Draw labels on batch processed frames with white text and black background (no confidence).
        
        frame: numpy BGR image (modified in-place)
        res: single Results object from YOLO model
        """
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return
        
        # Get box coordinates
        xyxy = getattr(res.boxes, 'xyxy', None)
        if xyxy is None:
            return
        
        if hasattr(xyxy, 'cpu'):
            boxes_arr = xyxy.cpu().numpy()
        else:
            boxes_arr = np.array(xyxy)
        
        # Get class IDs
        cls_ids = getattr(res.boxes, 'cls', None)
        if cls_ids is None:
            return
        if hasattr(cls_ids, 'cpu'):
            cls_arr = cls_ids.cpu().numpy()
        else:
            cls_arr = np.array(cls_ids)

        ids = getattr(res.boxes, 'id', None)
        if ids is not None:
            if hasattr(ids, 'cpu'):
                id_arr = ids.cpu().numpy()
            else:
                id_arr = np.array(ids)
        else:
            id_arr = [None] * len(cls_arr)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        for i, box in enumerate(boxes_arr):
            try:
                x1, y1, x2, y2 = map(int, box[:4])
            except Exception:
                continue
            
            cls_id = int(cls_arr[i])
            name = res.names.get(cls_id, str(cls_id))
            label = name
            if hasattr(self, 'batch_tracking_var') and self.batch_tracking_var.get():
                tid = None
                if id_arr is not None and len(id_arr) > i:
                    try:
                        tid = int(id_arr[i])
                    except Exception:
                        tid = None
                if tid is not None and tid != -1:
                    label = f"ID {tid} | {label}"
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label above box
            text_x = x1
            text_y = y1 - 6
            
            # If label would go above image, place it below
            if text_y - text_h - baseline < 0:
                text_y = y1 + text_h + 6
            
            # Draw black background rectangle
            rect_tl = (text_x, text_y - text_h - baseline)
            rect_br = (text_x + text_w, text_y + baseline)
            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
            
            # Draw white text
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    
    def _draw_batch_labels_with_font(self, frame, res, font_scale, show_tracking):
        """Draw labels on batch processed frames using font size from slider.
        
        frame: numpy BGR image (modified in-place)
        res: single Results object from YOLO model
        """
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return
        
        # Get box coordinates
        xyxy = getattr(res.boxes, 'xyxy', None)
        if xyxy is None:
            return
        
        if hasattr(xyxy, 'cpu'):
            boxes_arr = xyxy.cpu().numpy()
        else:
            boxes_arr = np.array(xyxy)
        
        # Get class IDs
        cls_ids = getattr(res.boxes, 'cls', None)
        if cls_ids is None:
            return
        if hasattr(cls_ids, 'cpu'):
            cls_arr = cls_ids.cpu().numpy()
        else:
            cls_arr = np.array(cls_ids)

        ids = getattr(res.boxes, 'id', None)
        if ids is not None:
            if hasattr(ids, 'cpu'):
                id_arr = ids.cpu().numpy()
            else:
                id_arr = np.array(ids)
        else:
            id_arr = [None] * len(cls_arr)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.2, float(font_scale))
        thickness = max(1, int(round(font_scale * 2)))
        
        for i, box in enumerate(boxes_arr):
            try:
                x1, y1, x2, y2 = map(int, box[:4])
            except Exception:
                continue
            
            cls_id = int(cls_arr[i])
            name = res.names.get(cls_id, str(cls_id))
            label = name
            if show_tracking:
                tid = None
                if id_arr is not None and len(id_arr) > i:
                    try:
                        tid = int(id_arr[i])
                    except Exception:
                        tid = None
                if tid is not None and tid != -1:
                    label = f"ID {tid} | {label}"
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label above box
            text_x = x1
            text_y = y1 - 6
            
            # If label would go above image, place it below
            if text_y - text_h - baseline < 0:
                text_y = y1 + text_h + 6
            
            # Draw black background rectangle
            rect_tl = (text_x, text_y - text_h - baseline)
            rect_br = (text_x + text_w, text_y + baseline)
            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
            
            # Draw white text
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    
    def batch_process_directory(self):
        """Open directory dialog and start batch processing"""
        if not self.models:
            messagebox.showwarning("Warning", "Please load at least one model first")
            return
        
        directory = filedialog.askdirectory(title="Select directory containing videos")
        if not directory:
            return
        
        # Get video files from directory
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        video_files = [f for f in os.listdir(directory) 
                      if os.path.isfile(os.path.join(directory, f)) 
                      and f.lower().endswith(video_extensions)]
        
        if not video_files:
            messagebox.showwarning("Warning", f"No video files found in {directory}")
            return
        
        # Start batch processing in a separate thread
        self.batch_processing = True
        self.compare_var.set(False)  # Disable comparison mode
        self.mix_var.set(False)
        self.compare_checkbox.config(state=tk.DISABLED)
        self.mix_checkbox.config(state=tk.DISABLED)
        self._update_class_checkboxes()
        self.batch_cancel_btn.config(state=tk.NORMAL)
        parallel_count = self.parallel_count_var.get()
        active_model = self.models[self.active_model_idx]['model']
        self._batch_settings = {
            'model': active_model,
            'confidence': float(self.conf_var.get()),
            'tracking': bool(self.batch_tracking_var.get()),
            'auto_save': bool(self.batch_autosave_var.get()),
            'font_scale': float(self.font_size_var.get()),
            'selected_names': {
                self.class_names[index]
                for index in self.selected_classes
                if index in self.class_names
            },
        }
        self._schedule_batch_slideshow()
        
        thread = threading.Thread(target=self._batch_process_worker, 
                                 args=(directory, video_files, parallel_count),
                                 daemon=True)
        thread.start()
    
    def _batch_process_worker(self, directory, video_files, parallel_count):
        """Worker thread for batch processing videos"""
        try:
            total_videos = len(video_files)
            processed_count = 0
            
            # Initialize slideshow
            with self.batch_preview_lock:
                self.batch_preview_frames = {}
                self.active_batch_videos = []
                self.batch_video_names = list(video_files)
                self.batch_current_video_idx = 0
            
            # Update UI
            self._update_batch_progress(0, total_videos, "Initializing batch processing...")
            
            # Create thread pool for parallel processing
            self.batch_executor = ThreadPoolExecutor(max_workers=parallel_count)
            futures = {}
            
            # Submit all tasks
            for video_file in video_files:
                if not self.batch_processing:
                    break
                
                video_path = os.path.join(directory, video_file)
                future = self.batch_executor.submit(self._process_single_video, 
                                                   video_path, video_file,
                                                   self._batch_settings)
                futures[future] = video_file
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                if not self.batch_processing:
                    break
                
                try:
                    result = future.result()
                    processed_count += 1
                    status = f"Processed: {processed_count}/{total_videos} - {result}"
                    self._update_batch_progress(processed_count, total_videos, status)
                except Exception as e:
                    processed_count += 1
                    status = f"Error processing {futures[future]}: {str(e)}"
                    self._update_batch_progress(processed_count, total_videos, status)
            
            if self.batch_processing:
                self._update_batch_progress(total_videos, total_videos, 
                                          f"Batch processing complete! {processed_count}/{total_videos} videos processed")
            else:
                self._update_batch_progress(processed_count, total_videos, 
                                          f"Batch processing cancelled. {processed_count}/{total_videos} videos processed")
        
        except Exception as e:
            self._update_batch_progress(0, len(video_files), f"Error: {str(e)}")
        
        finally:
            self.batch_processing = False
            if self.batch_executor:
                self.batch_executor.shutdown(wait=False)
            self.batch_progress_queue.put(('finished',))
    
    def _process_single_video(self, video_path, video_filename, settings):
        """Process a single video file with the active model"""
        with self.batch_preview_lock:
            if video_filename not in self.active_batch_videos:
                self.active_batch_videos.append(video_filename)
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return f"Failed to open {video_filename}"
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            writer = None
            if settings['auto_save']:
                source_dir = os.path.dirname(video_path)
                processed_dir = os.path.join(source_dir, 'processed')
                os.makedirs(processed_dir, exist_ok=True)
                name_without_ext = os.path.splitext(video_filename)[0]
                output_path = os.path.join(processed_dir, f"{name_without_ext}_processed.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    return f"Failed to create writer for {video_filename}"
            
            # Use settings captured on the GUI thread before the worker starts.
            model = settings['model']
            conf_thresh = settings['confidence']
            selected_names = settings['selected_names']
            tracking_enabled = settings['tracking']
            
            frame_count = 0
            preview_interval = 2  # Update grid preview frequently
            frame_idx = 0
            
            while self.batch_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Single model mode for batch processing
                    task = getattr(model, 'task', '')
                    is_semantic = (task == 'semantic')
                    if tracking_enabled and not is_semantic:
                        results = model.track(frame, verbose=False, persist=True, tracker="botsort.yaml")
                    else:
                        results = model(frame, verbose=False)
                    result = results[0] if isinstance(results, (list, tuple)) else results
                    
                    # Filter by selected classes and confidence
                    has_boxes = getattr(result, 'boxes', None) is not None
                    if has_boxes:
                        self._filter_result_by_names(
                            result, model, selected_names, conf_thresh
                        )
                    
                    plot_masks = getattr(result, 'masks', None) is not None or getattr(result, 'semantic_mask', None) is not None
                    plot_boxes = has_boxes and not is_semantic and not plot_masks
                    
                    # Plot results with dynamic drawing
                    annotated_frame = result.plot(boxes=plot_boxes, masks=plot_masks, labels=False, conf=False, line_width=2)
                    
                    # Apply custom class colors (e.g. green for tactile-tiles)
                    self._apply_custom_mask_colors(annotated_frame, result)
                    
                    self._draw_batch_labels_with_font(
                        annotated_frame,
                        result,
                        settings['font_scale'],
                        tracking_enabled,
                    )
                    
                    # Capture frame for slideshow preview
                    if frame_idx % preview_interval == 0:
                        with self.batch_preview_lock:
                            self.batch_preview_frames[video_filename] = annotated_frame.copy()
                    
                    if writer is not None:
                        writer.write(annotated_frame)
                    frame_count += 1
                    frame_idx += 1
                
                except Exception as e:
                    if writer is not None:
                        writer.write(frame)
                    frame_count += 1
                    frame_idx += 1
            
            # Release resources
            cap.release()
            if writer is not None:
                writer.release()
            
            with self.batch_preview_lock:
                if video_filename in self.active_batch_videos:
                    self.active_batch_videos.remove(video_filename)
                
            if frame_count > 0:
                saved = "saved" if writer is not None else "preview only"
                return f"✓ {video_filename} ({frame_count} frames, {saved})"
            else:
                return f"✗ {video_filename} (no frames processed)"
        
        except Exception as e:
            with self.batch_preview_lock:
                if video_filename in self.active_batch_videos:
                    self.active_batch_videos.remove(video_filename)
            return f"✗ {video_filename} - {str(e)}"
    
    def _schedule_batch_slideshow(self):
        """Schedule the real-time batch preview grid (up to 4 videos)"""
        if not self.batch_processing:
            return
        
        def update_grid():
            try:
                active_frames = []
                with self.batch_preview_lock:
                    for v_name in self.active_batch_videos[:4]:
                        if v_name in self.batch_preview_frames:
                            active_frames.append((v_name, self.batch_preview_frames[v_name].copy()))
                
                if not active_frames:
                    self.root.after(33, update_grid)
                    return
                
                # We need to construct a grid (up to 2x2)
                max_w_per_frame = 640
                max_h_per_frame = 360
                
                n = len(active_frames)
                cols = 2 if n > 1 else 1
                rows = 2 if n > 2 else 1
                
                combined_w = cols * max_w_per_frame
                combined_h = rows * max_h_per_frame
                combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
                
                for i, (v_name, frame) in enumerate(active_frames):
                    h, w = frame.shape[:2]
                    scale = min(max_w_per_frame / w, max_h_per_frame / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (new_w, new_h))
                    
                    x_offset = (i % 2) * max_w_per_frame
                    y_offset = (i // 2) * max_h_per_frame
                    
                    # Center in tile
                    x_start = x_offset + (max_w_per_frame - new_w) // 2
                    y_start = y_offset + (max_h_per_frame - new_h) // 2
                    
                    combined_frame[y_start:y_start+new_h, x_start:x_start+new_w] = resized
                    
                    # Add overlay label
                    cv2.putText(combined_frame, f"Processing: {v_name}", (x_start + 10, y_start + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                self._show_rgb_frame(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                pass
            
            # Schedule next update in ~33ms (real-time 30FPS)
            if self.batch_processing:
                self.root.after(33, update_grid)
        
        # Start immediately
        self.root.after(33, update_grid)
    
    def _update_batch_progress(self, current, total, status_text):
        """Pass background progress to the Tk event loop."""
        self.batch_progress_queue.put(('progress', current, total, status_text))

    def _poll_batch_progress(self):
        """Apply queued batch updates from the Tk main thread."""
        try:
            while True:
                event = self.batch_progress_queue.get_nowait()
                if event[0] == 'progress':
                    _, current, total, status_text = event
                    if total > 0:
                        self.batch_progress_bar['value'] = int((current / total) * 100)
                    self.batch_progress_label.config(text=status_text)
                elif event[0] == 'finished':
                    self.batch_cancel_btn.config(state=tk.DISABLED)
                    self.compare_checkbox.config(state=tk.NORMAL)
                    self.mix_checkbox.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self.root.after(100, self._poll_batch_progress)
    
    def cancel_batch_processing(self):
        """Cancel ongoing batch processing"""
        self.batch_processing = False
        self._update_batch_progress(0, 100, "Cancelling batch processing...")
        if self.batch_executor:
            self.batch_executor.shutdown(wait=False)

# ---------------------------
# Parameters
# ---------------------------
save_output = True
output_video = "output_segmentation.mp4"
max_display_width = 1280   # max width for display
gui = None
cap = None
out = None

def main():
    global gui, cap, out
    gui = DetectionGUI()
    gui.update()
    gui.root.mainloop()

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if save_output and out is not None:
        try:
            out.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == '__main__':
    main()
