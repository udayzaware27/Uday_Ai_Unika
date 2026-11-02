import cv2


class LineCrossingCounter:
    """Count tracked detections when their center crosses the vertical midpoint."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_sides = {}
        self.counts = {}
        self._fallback_tracks = {}
        self._next_fallback_id = 0
        self._frame_index = 0

    def update(self, result, model_names=None):
        """Update cumulative per-class counts and return them plus line position.

        Tracking IDs are required so one object is counted once per crossing.
        If the tracker has not assigned IDs yet, the current totals are kept.
        """
        if result is None:
            return dict(self.counts), None

        frame = getattr(result, 'orig_img', None)
        if frame is None:
            return dict(self.counts), None
        height, width = frame.shape[:2]
        line_x = width // 2
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return dict(self.counts), line_x

        coords = getattr(boxes, 'xyxy', None)
        classes = getattr(boxes, 'cls', None)
        ids = getattr(boxes, 'id', None)
        if coords is None or classes is None:
            return dict(self.counts), line_x

        names = model_names or getattr(result, 'names', {}) or {}
        class_values = names.values() if isinstance(names, dict) else names
        for class_name in class_values:
            label = _canonicalize_label(class_name)
            if label:
                self.counts.setdefault(label, 0)
        try:
            count = min(len(coords), len(classes))
        except Exception:
            return dict(self.counts), line_x

        self._frame_index += 1
        detections = []
        for index in range(count):
            try:
                box = coords[index].tolist()
                center_x = (float(box[0]) + float(box[2])) / 2.0
                class_id = int(classes[index].item())
                side = -1 if center_x < line_x else 1
                if isinstance(names, dict):
                    class_name = names.get(class_id, class_id)
                elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                    class_name = names[class_id]
                else:
                    class_name = class_id
                label = _canonicalize_label(class_name)
                if not label:
                    continue
                track_id = None
                if ids is not None and index < len(ids):
                    try:
                        track_id = int(ids[index].item())
                    except Exception:
                        pass
                detections.append((track_id, label, center_x, side))
            except Exception:
                continue

        # If the tracker has no IDs, associate detections by nearest center so
        # line counting still works while tracker IDs warm up or are unavailable.
        used_fallback = set()
        max_distance = max(60.0, width * 0.15)
        for track_id, label, center_x, side in detections:
            if track_id is None:
                candidates = [
                    (abs(center_x - track['x']), key, track)
                    for key, track in self._fallback_tracks.items()
                    if key not in used_fallback
                    and track['label'] == label
                    and self._frame_index - track['frame'] <= 8
                    and abs(center_x - track['x']) <= max_distance
                ]
                if candidates:
                    _, key, previous_track = min(candidates, key=lambda item: item[0])
                else:
                    key = self._next_fallback_id
                    self._next_fallback_id += 1
                    previous_track = None
                used_fallback.add(key)
                previous_side = previous_track['side'] if previous_track else None
                self._fallback_tracks[key] = {
                    'x': center_x, 'side': side, 'label': label,
                    'frame': self._frame_index,
                }
            else:
                previous_side = self.previous_sides.get(track_id)
                self.previous_sides[track_id] = side

            if previous_side is not None and previous_side != side:
                self.counts[label] = self.counts.get(label, 0) + 1

        stale_ids = [
            key for key, track in self._fallback_tracks.items()
            if self._frame_index - track['frame'] > 8
        ]
        for key in stale_ids:
            del self._fallback_tracks[key]

        return dict(self.counts), line_x


def draw_counting_line(frame, line_x=None):
    """Draw the vertical counting line at the frame midpoint."""
    if frame is None:
        return frame
    x = frame.shape[1] // 2 if line_x is None else int(line_x)
    cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), (0, 255, 255), 2, cv2.LINE_AA)
    text_x = max(8, min(frame.shape[1] - 180, x + 8))
    cv2.putText(frame, 'COUNTING LINE', (text_x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def _canonicalize_label(name):
    """Normalize raw model names such as 'no-helmet', 'no_helmet', or 'without helmet'."""
    if name is None:
        return ""
    text = str(name).strip().lower()
    text = text.replace('_', ' ').replace('-', ' ')
    text = ' '.join(text.split())
    if not text:
        return ""

    aliases = {
        'person': 'Person',
        'helmet': 'Helmet',
        'with helmet': 'Helmet',
        'with helmet person': 'Helmet',
        'no helmet': 'No Helmet',
        'without helmet': 'No Helmet',
        'not wearing helmet': 'No Helmet',
        'no helmet person': 'No Helmet',
        'vest': 'Vest',
        'with vest': 'Vest',
        'no vest': 'No Vest',
        'without vest': 'No Vest',
        'not wearing vest': 'No Vest',
    }
    if text in aliases:
        return aliases[text]

    if 'helmet' in text and 'no' in text:
        return 'No Helmet'
    if 'helmet' in text and 'without' in text:
        return 'No Helmet'
    if 'vest' in text and 'no' in text:
        return 'No Vest'
    if 'vest' in text and 'without' in text:
        return 'No Vest'
    if 'helmet' in text:
        return 'Helmet'
    if 'vest' in text:
        return 'Vest'
    return text.title()


def normalize_class_label(name):
    """Convert model labels into a clean display label."""
    return _canonicalize_label(name)


def summarize_model_counts(result, model_names=None):
    """Return counts for every class in the model, including zero-count classes."""
    if result is None:
        return {}

    names = model_names or getattr(result, 'names', {}) or {}
    ordered_names = {}
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            try:
                ordered_names[int(class_id)] = str(class_name)
            except Exception:
                ordered_names[str(class_id)] = str(class_name)
    else:
        ordered_names = {}

    counts = {}
    for class_id, class_name in ordered_names.items():
        label = _canonicalize_label(class_name)
        if label:
            counts[label] = 0

    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return counts

    cls_values = getattr(boxes, 'cls', None)
    if cls_values is None:
        return counts

    for index in range(len(cls_values)):
        try:
            cls_item = cls_values[index]
            cls_id = int(float(cls_item.item())) if hasattr(cls_item, 'item') else int(float(cls_item))
        except Exception:
            continue

        label = _canonicalize_label(ordered_names.get(cls_id, cls_id))
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    ordered_counts = {}
    for label in ['Person', 'Helmet', 'No Helmet', 'Vest', 'No Vest']:
        if label in counts:
            ordered_counts[label] = counts[label]
    for label, value in counts.items():
        if label not in ordered_counts:
            ordered_counts[label] = value
    return ordered_counts


def draw_count_summary_overlay(frame, summary, title="Model Counts"):
    """Draw a top-right box with the class counts for the active model."""
    if frame is None or not summary:
        return frame

    lines = [f"{label}: {count}" for label, count in summary.items()]
    if not lines:
        return frame

    h, w = frame.shape[:2]
    line_gap = 22
    title_height = 24
    max_line_len = max(len(line) for line in lines)
    box_w = min(360, max_line_len * 7 + 54)
    box_h = min(320, title_height + 26 + len(lines) * line_gap)
    x1 = max(18, w - box_w - 20)
    y1 = 18
    x2 = w - 20
    y2 = min(h - 18, y1 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(
        overlay,
        title,
        (x1 + 12, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for index, line in enumerate(lines):
        text_y = y1 + 48 + index * line_gap
        cv2.putText(
            overlay,
            line,
            (x1 + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
