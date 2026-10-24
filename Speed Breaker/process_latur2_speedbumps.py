#!/usr/bin/env python3
"""Detect speed bumps in the Latur 2 survey videos.

Reads latur2.xlsx, runs best (3).pt (road classes plus Speed-Bump) on each
video, and writes a CSV row plus a screenshot whenever a speed bump is seen.
Latitude and longitude come from the GoPro GPS track embedded in the file.
"""

import argparse
import csv
import math
import struct
import subprocess
import sys
import threading
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "latur2.xlsx"
MODEL_PATH = ROOT / "best (3).pt"
OUT_DIR = ROOT / "latur2_output"
SHOT_DIR = OUT_DIR / "screenshots"
CSV_PATH = OUT_DIR / "speed_bumps.csv"
DONE_PATH = OUT_DIR / "processed_videos.txt"

FRAME_W = 1920
FRAME_H = 1080
SAMPLE_FPS = 3
INFER_BATCH = 8
SPEED_BUMP_CONF = 0.40
OTHER_CONF = 0.25
EVENT_GAP_SEC = 1.2
MERGE_METERS = 10.0
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

CSV_FIELDS = [
    "road_no",
    "start_chainage",
    "end_chainage",
    "estimated_chainage",
    "sub_division",
    "junior_executive",
    "surveyor",
    "file_name",
    "video_url",
    "survey_date",
    "video_time_sec",
    "gps_time_utc",
    "latitude",
    "longitude",
    "altitude_m",
    "speed_kmh",
    "confidence",
    "bbox",
    "other_detections",
    "screenshot",
]


def log(msg):
    print(msg, flush=True)


def load_sheet(path):
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        strings = []
        for si in root.findall("m:si", NS):
            strings.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.findall("m:sheetData/m:row", NS):
        values = {}
        for cell in row.findall("m:c", NS):
            ref = cell.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            node = cell.find("m:v", NS)
            raw = node.text if node is not None else ""
            if cell.attrib.get("t") == "s" and raw != "":
                raw = strings[int(raw)]
            values[col] = raw
        if values:
            rows.append(values)
    if not rows:
        return []
    header = rows[0]
    records = []
    for values in rows[1:]:
        item = {header[col]: values.get(col, "") for col in header}
        if item.get("subDivision") == "Latur 2" and item.get("data_status[0].video_url"):
            records.append(item)
    return records


def parse_chainage(text):
    km, meters = str(text).split("/")
    return int(km) + int(meters) / 1000.0


def format_chainage(km_value):
    negative = km_value < 0
    km_value = abs(km_value)
    km = int(km_value)
    meters = int(round((km_value - km) * 1000))
    if meters == 1000:
        km += 1
        meters = 0
    text = f"{km}/{meters:03d}"
    return f"-{text}" if negative else text


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def iter_klv(buf):
    index = 0
    total = len(buf)
    while index + 8 <= total:
        key = buf[index : index + 4]
        if len(key) < 4 or not all(32 <= byte < 127 for byte in key):
            break
        typ = buf[index + 4]
        sample_size = buf[index + 5]
        repeat = int.from_bytes(buf[index + 6 : index + 8], "big")
        payload_len = sample_size * repeat
        if payload_len < 0 or index + 8 + payload_len > total:
            break
        payload = buf[index + 8 : index + 8 + payload_len]
        yield key, typ, sample_size, repeat, payload
        pad = (4 - (payload_len % 4)) % 4
        index += 8 + payload_len + pad


def _gps_points_from_stream(children):
    gps = next((item for item in children if item[0] == b"GPS9"), None)
    if gps is None:
        gps = next((item for item in children if item[0] == b"GPS5"), None)
    if gps is None:
        return []
    scale_item = next((item for item in children if item[0] == b"SCAL"), None)
    _key, _typ, sample_size, repeat, payload = gps
    if scale_item is not None and len(scale_item[4]) >= 36:
        scales = struct.unpack(">9i", scale_item[4][:36])
    else:
        scales = (10_000_000, 10_000_000, 1000, 1000, 100, 1, 1000, 100, 1)
    points = []
    if sample_size < 32:
        return points
    for rep in range(repeat):
        sample = payload[rep * sample_size : (rep + 1) * sample_size]
        if len(sample) < 32:
            continue
        lat_i, lon_i, alt_i, spd2_i, spd3_i, days_i, secs_i, dop_i, fix_i = struct.unpack(">7i2h", sample[:32])
        fix = fix_i / scales[8] if scales[8] else fix_i
        if fix < 2:
            continue
        lat = lat_i / scales[0]
        lon = lon_i / scales[1]
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if abs(lat) < 0.01 and abs(lon) < 0.01:
            continue
        secs = secs_i / scales[6] if scales[6] else secs_i
        days = days_i / scales[5] if scales[5] else days_i
        gps_time = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=days, seconds=secs)
        points.append(
            {
                "lat": lat,
                "lon": lon,
                "alt": alt_i / scales[2] if scales[2] else alt_i,
                "speed_kmh": (spd2_i / scales[3] if scales[3] else spd2_i) * 3.6,
                "gps_time": gps_time,
            }
        )
    return points


def extract_gps_points(packet):
    found = []

    def walk(buf):
        children = list(iter_klv(buf))
        if any(item[0] in (b"GPS9", b"GPS5") for item in children):
            found.extend(_gps_points_from_stream(children))
            return
        for key, typ, _ss, _rep, payload in children:
            if typ == 0 and key in (b"DEVC", b"STRM", b"GPMF"):
                walk(payload)

    walk(packet)
    return found


def read_framecrc(path):
    packets = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        pts = int(parts[2])
        duration = int(parts[3])
        size = int(parts[4])
        packets.append((pts / 1000.0, duration / 1000.0, size))
    return packets


def load_gps_track(gpmd_path, crc_path):
    packets = read_framecrc(crc_path)
    blob = Path(gpmd_path).read_bytes()
    track = []
    offset = 0
    for index, (pts, duration, size) in enumerate(packets):
        chunk = blob[offset : offset + size]
        offset += size
        if len(chunk) < size:
            break
        points = extract_gps_points(chunk)
        if not points:
            continue
        next_pts = packets[index + 1][0] if index + 1 < len(packets) else pts + max(duration, 0.1 * len(points))
        span = max(0.05, next_pts - pts)
        step = span / max(1, len(points))
        for sample_index, point in enumerate(points):
            point = dict(point)
            point["t"] = pts + sample_index * step
            track.append(point)
    track.sort(key=lambda item: item["t"])
    return track


def interpolate_gps(track, timestamp):
    if not track:
        return None
    if timestamp <= track[0]["t"]:
        return dict(track[0]) if track[0]["t"] - timestamp <= 2 else None
    if timestamp >= track[-1]["t"]:
        return dict(track[-1]) if timestamp - track[-1]["t"] <= 2 else None
    lo, hi = 0, len(track) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if track[mid]["t"] <= timestamp:
            lo = mid
        else:
            hi = mid
    left, right = track[lo], track[hi]
    span = right["t"] - left["t"]
    frac = 0.0 if span <= 0 else (timestamp - left["t"]) / span
    gps_time = left["gps_time"] + (right["gps_time"] - left["gps_time"]) * frac
    return {
        "t": timestamp,
        "lat": left["lat"] + (right["lat"] - left["lat"]) * frac,
        "lon": left["lon"] + (right["lon"] - left["lon"]) * frac,
        "alt": left["alt"] + (right["alt"] - left["alt"]) * frac,
        "speed_kmh": left["speed_kmh"] + (right["speed_kmh"] - left["speed_kmh"]) * frac,
        "gps_time": gps_time,
    }


def track_distance_table(track):
    if not track:
        return []
    cumulative = [0.0]
    for prev, curr in zip(track, track[1:]):
        cumulative.append(cumulative[-1] + haversine_m(prev["lat"], prev["lon"], curr["lat"], curr["lon"]))
    return cumulative


def chainage_at(track, cumulative, timestamp, start_km, end_km, duration):
    if track and cumulative and cumulative[-1] > 1:
        if timestamp <= track[0]["t"]:
            frac = 0.0
        elif timestamp >= track[-1]["t"]:
            frac = 1.0
        else:
            lo, hi = 0, len(track) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if track[mid]["t"] <= timestamp:
                    lo = mid
                else:
                    hi = mid
            span = track[hi]["t"] - track[lo]["t"]
            local = 0.0 if span <= 0 else (timestamp - track[lo]["t"]) / span
            travelled = cumulative[lo] + (cumulative[hi] - cumulative[lo]) * local
            frac = travelled / cumulative[-1]
    elif duration > 0:
        frac = min(1.0, max(0.0, timestamp / duration))
    else:
        return ""
    return format_chainage(start_km + frac * (end_km - start_km))


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_video(url, gpmd_path, crc_path, seconds=None, seek=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if seek:
        cmd += ["-ss", str(seek)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += [
        "-i",
        url,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H}:flags=bilinear,format=bgr24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
        "-map",
        "0:3",
        "-c",
        "copy",
        "-f",
        "data",
        str(gpmd_path),
        "-map",
        "0:3",
        "-c",
        "copy",
        "-f",
        "framecrc",
        str(crc_path),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def boxes_of(result, names):
    found = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return found
    xyxy = boxes.xyxy.tolist()
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    for coords, cls_id, conf in zip(xyxy, cls_ids, confs):
        found.append(
            {
                "name": names[int(cls_id)],
                "conf": float(conf),
                "bbox": [round(float(v), 1) for v in coords],
            }
        )
    return found


def detection_summary(items):
    counts = Counter(item["name"] for item in items if item["name"] != "Speed-Bump")
    return ";".join(f"{name}:{counts[name]}" for name in sorted(counts))


def stamp_screenshot(image, row):
    lines = [
        f"Speed-Bump {float(row['confidence']):.2f}",
        f"{row['latitude']}, {row['longitude']}" if row["latitude"] else "GPS unavailable",
        f"{row['road_no']}  {row['estimated_chainage'] or row['start_chainage']}",
        f"{row['file_name']}  t={float(row['video_time_sec']):.1f}s",
    ]
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (760, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    y = 42
    for line in lines:
        cv2.putText(image, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return image


def merge_events(events):
    merged = []
    for event in sorted(events, key=lambda item: item["t"]):
        if not merged:
            merged.append(event)
            continue
        prev = merged[-1]
        same_place = False
        if prev.get("lat") is not None and event.get("lat") is not None:
            same_place = haversine_m(prev["lat"], prev["lon"], event["lat"], event["lon"]) <= MERGE_METERS
        elif event["t"] - prev["t"] <= 4.0:
            same_place = True
        if same_place:
            if event["conf"] > prev["conf"]:
                merged[-1] = event
            continue
        merged.append(event)
    return merged


def open_csv():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    handle = CSV_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        handle.flush()
    return handle, writer


def done_videos():
    if not DONE_PATH.exists():
        return set()
    return {line.strip() for line in DONE_PATH.read_text().splitlines() if line.strip()}


def mark_done(file_name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DONE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(file_name + "\n")


def process_video(model, names, record, writer, csv_handle, seconds=None, seek=None):
    url = record["data_status[0].video_url"]
    file_name = record["data_status[0].fileName"]
    stem = Path(file_name).stem
    work = OUT_DIR / "_work" / stem
    work.mkdir(parents=True, exist_ok=True)
    gpmd_path = work / "gpmd.bin"
    crc_path = work / "gpmd.crc"
    for path in (gpmd_path, crc_path):
        if path.exists():
            path.unlink()

    log(f"Starting {file_name}")
    proc = decode_video(url, gpmd_path, crc_path, seconds=seconds, seek=seek)
    stderr_lines = []

    def drain():
        stderr_lines.append(proc.stderr.read().decode("utf-8", errors="replace"))

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()

    frame_bytes = FRAME_W * FRAME_H * 3
    batch = []
    times = []
    frame_index = 0
    class_hits = Counter()
    events = []
    current = None
    start_km = parse_chainage(record["startChainage"])
    end_km = parse_chainage(record["endChainage"])

    def flush_current():
        nonlocal current
        if current is not None:
            events.append(current)
            current = None

    def consume(frames, frame_times):
        nonlocal current
        results = model.predict(
            frames,
            imgsz=1280,
            conf=OTHER_CONF,
            verbose=False,
            device=0,
        )
        for frame, timestamp, result in zip(frames, frame_times, results):
            items = boxes_of(result, names)
            for item in items:
                class_hits[item["name"]] += 1
            bumps = [item for item in items if item["name"] == "Speed-Bump" and item["conf"] >= SPEED_BUMP_CONF]
            if not bumps:
                if current is not None and timestamp - current["last_t"] > EVENT_GAP_SEC:
                    flush_current()
                continue
            best = max(bumps, key=lambda item: item["conf"])
            annotated = result.plot(line_width=2, conf=True, labels=True)
            if current is None or timestamp - current["last_t"] > EVENT_GAP_SEC:
                flush_current()
                current = {
                    "t": timestamp,
                    "last_t": timestamp,
                    "conf": best["conf"],
                    "bbox": best["bbox"],
                    "other": detection_summary(items),
                    "image": annotated,
                }
            else:
                current["last_t"] = timestamp
                if best["conf"] > current["conf"]:
                    current.update(
                        {
                            "t": timestamp,
                            "conf": best["conf"],
                            "bbox": best["bbox"],
                            "other": detection_summary(items),
                            "image": annotated,
                        }
                    )

    while True:
        raw = read_exact(proc.stdout, frame_bytes)
        if raw is None:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3)
        timestamp = frame_index / SAMPLE_FPS
        if seek:
            timestamp += float(seek)
        batch.append(frame.copy())
        times.append(timestamp)
        frame_index += 1
        if len(batch) >= INFER_BATCH:
            consume(batch, times)
            batch = []
            times = []
        if frame_index % 90 == 0:
            log(f"  {file_name}: {frame_index} frames ({timestamp:.0f}s), speed-bump groups {len(events)}")

    if batch:
        consume(batch, times)
    flush_current()
    proc.stdout.close()
    return_code = proc.wait()
    drain_thread.join(timeout=5)
    if return_code != 0:
        log(f"ffmpeg failed for {file_name}: {''.join(stderr_lines)[-500:]}")
        return class_hits

    track = load_gps_track(gpmd_path, crc_path)
    cumulative = track_distance_table(track)
    duration = frame_index / SAMPLE_FPS
    log(f"  {file_name}: decoded {frame_index} frames, GPS samples {len(track)}, raw groups {len(events)}")

    located = []
    for event in events:
        gps = interpolate_gps(track, event["t"])
        event = dict(event)
        if gps:
            event.update({"lat": gps["lat"], "lon": gps["lon"], "alt": gps["alt"], "speed_kmh": gps["speed_kmh"], "gps_time": gps["gps_time"]})
        else:
            event.update({"lat": None, "lon": None, "alt": None, "speed_kmh": None, "gps_time": None})
        located.append(event)
    located = merge_events(located)

    shot_dir = SHOT_DIR / stem
    shot_dir.mkdir(parents=True, exist_ok=True)
    for event in located:
        chainage = chainage_at(track, cumulative, event["t"], start_km, end_km, duration)
        if event["lat"] is None:
            lat_text = ""
            lon_text = ""
            alt_text = ""
            speed_text = ""
            gps_text = ""
            shot_name = f"{stem}_t{event['t']:.1f}_nogps.jpg"
        else:
            lat_text = f"{event['lat']:.7f}"
            lon_text = f"{event['lon']:.7f}"
            alt_text = f"{event['alt']:.1f}"
            speed_text = f"{event['speed_kmh']:.1f}"
            gps_text = event["gps_time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            shot_name = f"{stem}_t{event['t']:.1f}_{lat_text}_{lon_text}.jpg"
        row = {
            "road_no": record["roadNo"],
            "start_chainage": record["startChainage"],
            "end_chainage": record["endChainage"],
            "estimated_chainage": chainage,
            "sub_division": record["subDivision"],
            "junior_executive": record["juniorExecutiveName"],
            "surveyor": record["surveyorName"],
            "file_name": file_name,
            "video_url": url,
            "survey_date": record["Date"],
            "video_time_sec": f"{event['t']:.2f}",
            "gps_time_utc": gps_text,
            "latitude": lat_text,
            "longitude": lon_text,
            "altitude_m": alt_text,
            "speed_kmh": speed_text,
            "confidence": f"{event['conf']:.3f}",
            "bbox": " ".join(str(v) for v in event["bbox"]),
            "other_detections": event["other"],
            "screenshot": str((shot_dir / shot_name).relative_to(ROOT)),
        }
        image = stamp_screenshot(event["image"], row)
        cv2.imwrite(str(shot_dir / shot_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        writer.writerow(row)
    csv_handle.flush()
    log(f"Finished {file_name}: {len(located)} speed bumps saved")
    if not seconds and not seek:
        mark_done(file_name)
    counts_path = OUT_DIR / "class_counts.csv"
    write_header = not counts_path.exists()
    with counts_path.open("a", newline="", encoding="utf-8") as handle:
        count_writer = csv.writer(handle)
        if write_header:
            count_writer.writerow(["file_name", "frames", "class_name", "detection_hits"])
        for class_name, hits in sorted(class_hits.items()):
            count_writer.writerow([file_name, frame_index, class_name, hits])
    return class_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None, help="Process only the first N seconds of each video")
    parser.add_argument("--seek", type=float, default=None, help="Seek to this second before processing")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos")
    parser.add_argument("--force", action="store_true", help="Reprocess videos already present in the CSV")
    args = parser.parse_args()

    records = load_sheet(XLSX)
    if args.limit:
        records = records[: args.limit]
    if not records:
        log("No Latur 2 videos found in the sheet")
        return 1

    from ultralytics import YOLO

    log(f"Loading {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    names = model.names
    log(f"Classes: {names}")

    csv_handle, writer = open_csv()
    already = set() if args.force or args.seconds or args.seek else done_videos()
    try:
        for record in records:
            file_name = record["data_status[0].fileName"]
            if file_name in already:
                log(f"Skipping {file_name}; already in {CSV_PATH.name}")
                continue
            try:
                process_video(model, names, record, writer, csv_handle, seconds=args.seconds, seek=args.seek)
            except Exception as exc:
                log(f"Failed {file_name}: {exc}")
    finally:
        csv_handle.close()
    log(f"CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
