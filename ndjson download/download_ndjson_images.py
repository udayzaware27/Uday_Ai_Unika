#!/usr/bin/env python3
"""
Generic NDJSON image downloader.

Reads any NDJSON file containing image records with a URL field and downloads
the images into one output directory. It does not train a model or create labels.

Supported URL keys (checked in this order):
    url, image_url, image, download_url

Optional filename keys:
    file, filename, name

Examples:
    python3 download_ndjson_images.py /path/to/data.ndjson /path/to/output_images
    python3 download_ndjson_images.py ./data.ndjson ./images --workers 8
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


URL_KEYS = ("url", "image_url", "image", "download_url")
FILENAME_KEYS = ("file", "filename", "name")

# Set the paths and download settings directly in code.
NDJSON_PATH = "/home/aiunika/Downloads/Uday (2)/Uday/ndjson download/pcb-defects-detection.ndjson"
OUTPUT_PATH = "/home/aiunika/Downloads/Uday (2)/Uday/ndjson download/pcb images"
WORKERS = 8
TIMEOUT = 45
RETRIES = 3
OVERWRITE = False


def clean_filename(value, fallback):
    """Return a safe filename while preserving its extension when possible."""
    name = os.path.basename(str(value or "")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name or name in {".", ".."}:
        name = fallback
    if not Path(name).suffix:
        name += ".jpg"
    return name


def find_url(record):
    """Find an image URL in a JSON object."""
    for key in URL_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
            return value.strip()
    return None


def find_filename(record, url, index):
    """Choose a filename from the record or URL."""
    value = None
    for key in FILENAME_KEYS:
        candidate = record.get(key)
        if isinstance(candidate, str) and candidate.strip():
            value = candidate.strip()
            break

    if not value:
        value = Path(urlparse(url).path).name

    return clean_filename(value, f"image_{index:06d}.jpg")


def read_records(ndjson_path):
    """Read NDJSON records; ignore malformed lines and records without image URLs."""
    records = []
    with open(ndjson_path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON on line {line_number}")
                continue

            if not isinstance(record, dict):
                continue

            # Handle common nested structures as well as top-level URL fields.
            candidates = [record]
            for key in ("data", "image_data", "image"):
                nested = record.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested)

            url = None
            filename = None
            for candidate in candidates:
                url = find_url(candidate)
                if url:
                    filename = find_filename(candidate, url, line_number)
                    break

            if url:
                records.append((filename, url, line_number))

    # Make filenames unique, preventing records from overwriting each other.
    used = {}
    unique_records = []
    for filename, url, line_number in records:
        used[filename] = used.get(filename, 0) + 1
        if used[filename] > 1:
            stem, ext = os.path.splitext(filename)
            filename = f"{stem}_{used[filename]}{ext or '.jpg'}"
        unique_records.append((filename, url, line_number))

    return unique_records


def download_one(record, output_dir, timeout, retries, overwrite):
    filename, url, line_number = record
    destination = output_dir / filename

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return "skipped", filename, None

    request = urllib.request.Request(
        url, headers={"User-Agent": "NDJSON-Image-Downloader/1.0"}
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise urllib.error.HTTPError(
                        url, status, f"HTTP status {status}", response.headers, None
                    )
                content = response.read()

            if not content:
                raise OSError("Downloaded file was empty")

            temporary = destination.with_name(destination.name + ".part")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return "downloaded", filename, None

        except Exception as exc:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
            else:
                return "failed", filename, f"line {line_number}: {exc}"

    return "failed", filename, f"line {line_number}: unknown error"


def main():
    source = Path(NDJSON_PATH).expanduser()
    output_dir = Path(OUTPUT_PATH).expanduser()

    if not source.is_file():
        raise FileNotFoundError(f"Input NDJSON file does not exist: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(source)

    if not records:
        print(
            "No image URLs found. Expected one of these keys in each JSON record: "
            + ", ".join(URL_KEYS)
        )
        print("If your dataset uses a different schema, update URL_KEYS/find_url in the script.")
        return

    print(f"Input file: {source.resolve()}")
    print(f"Image URLs found: {len(records):,}")
    print(f"Output folder: {output_dir.resolve()}")
    print("Downloading images only;")

    downloaded = skipped = failed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(
                download_one, record, output_dir, TIMEOUT, RETRIES, OVERWRITE
            )
            for record in records
        ]
        total = len(futures)

        for completed, future in enumerate(as_completed(futures), start=1):
            status, filename, error = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(f"{filename}: {error}")

            if completed % 100 == 0 or completed == total:
                print(
                    f"Progress: {completed:,}/{total:,} | downloaded: {downloaded:,} | "
                    f"skipped: {skipped:,} | failed: {failed:,}",
                    flush=True
                )

    if failures:
        report = output_dir / "download_failures.txt"
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Failed-download report: {report.resolve()}")

    print("\nComplete.")
    print(f"Downloaded: {downloaded:,}")
    print(f"Skipped (already exist): {skipped:,}")
    print(f"Failed: {failed:,}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
