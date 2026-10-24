#!/usr/bin/env python3
"""
Unified NDJSON Dataset Processor.

Reads an Ultralytics-style NDJSON file, creates 'Images' and 'Labels' folders:
  - Downloads raw, un-annotated images into 'Images/'
  - Generates companion YOLO annotation files in 'Labels/' (<image_name>.txt)
  - Exports 'classes.txt' mapping numeric class IDs to class names
"""

# ===================== CONFIGURATION =====================
INPUT_NDJSON = "/home/aiunika/Downloads/Uday (2)/Uday/ndjson download/pcb-defects-detection.ndjson"
OUTPUT_DIR = "/home/aiunika/Downloads/Uday (2)/Uday/ndjson download/dataset"

WORKERS = 8
TIMEOUT = 45
RETRIES = 3
OVERWRITE = False
# =========================================================

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Install it with: python3 -m pip install Pillow")

URL_KEYS = ("url", "image_url", "download_url")


def get_url(record):
    for key in URL_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def read_dataset(path):
    """Parse NDJSON header metadata and image entries."""
    metadata = {}
    images = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON at line {line_no}; skipping")
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "dataset":
                metadata = item
            elif item.get("type") == "image" or (get_url(item) and "annotations" in item):
                url = get_url(item)
                if url:
                    images.append((line_no, item, url))
    return metadata, images


def get_class_names(metadata):
    """Extract class mapping from dataset metadata."""
    names = metadata.get("class_names", {}) if isinstance(metadata, dict) else {}
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        result = {}
        for key, value in names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def safe_filename(value, fallback):
    """Sanitize filename to avoid invalid path characters."""
    name = os.path.basename(str(value or "")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name or name in (".", ".."):
        name = fallback
    if not Path(name).suffix:
        name += ".jpg"
    return name


def parse_boxes(record):
    """
    Parse (class_id, xc, yc, w, h) from Ultralytics-style NDJSON annotations.
    Supports both list format: [class_id, xc, yc, w, h] and dictionary format.
    """
    annotations = record.get("annotations", {})
    boxes = annotations.get("boxes", []) if isinstance(annotations, dict) else []
    parsed = []
    for box in boxes or []:
        if isinstance(box, (list, tuple)) and len(box) >= 5:
            try:
                cls_id = int(box[0])
                xc, yc, bw, bh = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                if all(0 <= v <= 1 for v in (xc, yc, bw, bh)) and bw > 0 and bh > 0:
                    parsed.append((cls_id, xc, yc, bw, bh))
            except (TypeError, ValueError):
                continue
        elif isinstance(box, dict):
            try:
                cls_id = int(box.get("class_id", box.get("class", box.get("label_id", 0))))
                xc = float(box.get("x_center", box.get("xc")))
                yc = float(box.get("y_center", box.get("yc")))
                bw = float(box.get("width", box.get("w")))
                bh = float(box.get("height", box.get("h")))
                if all(0 <= v <= 1 for v in (xc, yc, bw, bh)) and bw > 0 and bh > 0:
                    parsed.append((cls_id, xc, yc, bw, bh))
            except (TypeError, ValueError):
                continue
    return parsed


def download_bytes(url, timeout=45, retries=3):
    request = urllib.request.Request(
        url, headers={"User-Agent": "NDJSON-Dataset-Pipeline/1.0"}
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise OSError("empty response received")
            return data
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def process_entry(entry, images_dir, labels_dir, timeout, retries, overwrite):
    """Downloads the image to Images/ and writes YOLO coordinates to Labels/."""
    line_no, record, url, filename = entry
    stem = Path(filename).stem
    image_dest = images_dir / filename
    label_dest = labels_dir / f"{stem}.txt"

    # 1. Write the YOLO label file
    boxes = parse_boxes(record)
    label_lines = [
        f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        for cls_id, xc, yc, bw, bh in boxes
    ]
    with open(label_dest, "w", encoding="utf-8") as lf:
        if label_lines:
            lf.write("\n".join(label_lines) + "\n")

    # 2. Download the original image
    if image_dest.exists() and image_dest.stat().st_size > 0 and not overwrite:
        return "skipped", filename, None

    try:
        data = download_bytes(url, timeout=timeout, retries=retries)
        temp_path = images_dir / (filename + ".part")
        temp_path.write_bytes(data)

        # Validate that the downloaded payload is a readable image and standardize to RGB
        with Image.open(temp_path) as im:
            im.verify()
        with Image.open(temp_path) as im:
            rgb = im.convert("RGB")
            rgb.save(image_dest, quality=95)

        temp_path.unlink(missing_ok=True)
        return "downloaded", filename, None
    except Exception as exc:
        return "failed", filename, f"line {line_no}: {exc}"


def main():
    source = Path(INPUT_NDJSON).expanduser()
    output_root = Path(OUTPUT_DIR).expanduser()

    if not source.is_file():
        raise SystemExit(f"NDJSON file not found: {source}")

    images_dir = output_root / "Images"
    labels_dir = output_root / "Labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    metadata, records = read_dataset(source)
    class_names = get_class_names(metadata)

    if not records:
        raise SystemExit("No image records with downloadable URLs were found in this NDJSON file.")

    # Save classes.txt if metadata is provided
    classes_file = output_root / "classes.txt"
    if class_names:
        with open(classes_file, "w", encoding="utf-8") as cf:
            for cls_id in sorted(class_names.keys()):
                cf.write(f"{class_names[cls_id]}\n")

    print(f"Dataset root: {output_root.resolve()}")
    print(f"Total image records: {len(records):,}")
    print(f"Classes found ({len(class_names)}): {class_names}")

    # Prepare uniquely deduplicated filenames
    used_names = {}
    prepared = []
    for line_no, record, url in records:
        original = record.get("file") or Path(urlparse(url).path).name
        filename = safe_filename(original, f"image_{line_no:06d}.jpg")
        used_names[filename] = used_names.get(filename, 0) + 1
        if used_names[filename] > 1:
            stem, ext = os.path.splitext(filename)
            filename = f"{stem}_{used_names[filename]}{ext or '.jpg'}"
        prepared.append((line_no, record, url, filename))

    downloaded = skipped = failed = 0
    failures = []

    print("\nDownloading images and generating label files...")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(process_entry, entry, images_dir, labels_dir, TIMEOUT, RETRIES, OVERWRITE)
            for entry in prepared
        ]
        total = len(futures)
        for n, future in enumerate(as_completed(futures), 1):
            status, filename, error = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(f"{filename}: {error}")

            if n % 100 == 0 or n == total:
                print(
                    f"Progress {n:,}/{total:,} | downloaded: {downloaded:,} | "
                    f"skipped: {skipped:,} | failed: {failed:,}",
                    flush=True,
                )

    if failures:
        report = output_root / "download_failures.txt"
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"\nDownload failures logged to: {report.resolve()}")

    print("\nFinished successfully.")
    print(f"- Clean images saved to: {images_dir.resolve()}")
    print(f"- YOLO labels saved to:  {labels_dir.resolve()}")
    if class_names:
        print(f"- Class definitions:    {classes_file.resolve()}")


if __name__ == "__main__":
    main()