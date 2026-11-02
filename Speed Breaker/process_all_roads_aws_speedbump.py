#!/usr/bin/env python3
"""Speed Bump Video Detection & Telemetry Processing.

Reads survey sheet, combines multi-part GoPro videos, runs YOLO speedbump detection,
extracts GPS tracks for chainage, stamps annotated frames, uploads to AWS S3,
and generates Excel & CSV reports.
"""

import argparse
import csv
import math
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import xml.sax.saxutils as sax
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_XLSX = SCRIPT_DIR / "rasta_ai.unprocessed-180.xlsx"
DEFAULT_MODEL = REPO_ROOT / "Models" / "speedbump.pt"
DEFAULT_OUT = SCRIPT_DIR / "reprocessed_runs"

DET_FIELDS = [
    "road_no", "road_name", "start_chainage", "end_chainage", "estimated_chainage", "sub_division",
    "data_status[0].video_url", "data_status[1].video_url", "data_status[0].fileName", "data_status[1].fileName",
    "speed_bump_index", "video_time_sec", "source_video_part", "gps_time_utc", "latitude", "longitude",
    "altitude_m", "speed_kmh", "confidence", "bbox", "google_map_link", "aws_s3_url", "local_screenshot", "survey_date"
]

SUM_FIELDS = [
    "road_no", "road_name", "start_chainage", "end_chainage", "sub_division", "distance_km",
    "data_status[0].video_url", "data_status[1].video_url", "speed_bumps_count", "status", "all_aws_frame_urls"
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# Excel (.xlsx) & Data Sheet Helpers
# ==============================================================================
def save_xlsx(path, sheet_name, headers, rows):
    col_str = lambda c: chr(c % 26 + 65) if c < 26 else chr(c // 26 - 1 + 65) + chr(c % 26 + 65)
    data = [f'<row r="1">{"".join(f"""<c r="{col_str(i)}1" t="inlineStr"><is><t>{sax.escape(str(h))}</t></is></c>""" for i, h in enumerate(headers))}</row>']
    for r_idx, r in enumerate(rows, start=2):
        data.append(f'<row r="{r_idx}">{"".join(f"""<c r="{col_str(i)}{r_idx}" t="inlineStr"><is><t>{sax.escape(str(v if v is not None else ""))}</t></is></c>""" for i, v in enumerate(r))}</row>')
    ws = f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(data)}</sheetData></worksheet>'
    wb = f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{sax.escape(sheet_name[:31])}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    ct = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    wb_rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", wb)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", ws)


def load_sheet(path, subdivision=None):
    with zipfile.ZipFile(path) as zf:
        s_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        strs = ["".join(t.text or "" for t in si.findall(".//{*}t")) for si in s_root.findall("{*}si")]
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.findall(".//{*}row"):
        cells = {("".join(ch for ch in c.attrib.get("r", "") if ch.isalpha())): (strs[int(c.find("{*}v").text)] if c.attrib.get("t") == "s" and c.find("{*}v") is not None and c.find("{*}v").text else (c.find("{*}v").text if c.find("{*}v") is not None else "")) for c in r.findall("{*}c")}
        if cells:
            rows.append(cells)
    if not rows:
        return []
    hdr, records = rows[0], []
    for r in rows[1:]:
        item = {hdr[c]: r.get(c, "") for c in hdr}
        if item.get("data_status[0].video_url") and (not subdivision or item.get("subDivision", "").strip().lower() == subdivision.strip().lower()):
            records.append(item)
    return records


# ==============================================================================
# Chainage & GoPro Telemetry (GPMF) Extraction
# ==============================================================================
def parse_chainage(text):
    if not text:
        return 0.0
    text, neg = str(text).strip(), str(text).strip().startswith("-")
    clean = text.lstrip("-")
    try:
        if "/" in clean:
            k, m = clean.split("/", 1)
            return (-1 if neg else 1) * (float(k) + float(m) / 1000.0)
        return (-1 if neg else 1) * float(clean)
    except ValueError:
        return 0.0


def format_chainage(km):
    neg, km = km < 0, abs(km)
    k, m = int(km), int(round((km - int(km)) * 1000))
    if m >= 1000:
        k, m = k + 1, 0
    return f"{'-' if neg else ''}{k}/{m:03d}"


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2, dlat, dlon = math.radians(lat1), math.radians(lat2), math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def iter_klv(buf):
    idx, total = 0, len(buf)
    while idx + 8 <= total:
        key, typ, sz, rep = buf[idx:idx+4], buf[idx+4], buf[idx+5], int.from_bytes(buf[idx+6:idx+8], "big")
        if not all(32 <= b < 127 for b in key) or idx + 8 + sz * rep > total:
            break
        yield key, typ, sz, rep, buf[idx+8 : idx+8 + sz*rep]
        idx += 8 + sz * rep + (4 - ((sz * rep) % 4)) % 4


def extract_gps_points(buf):
    found = []
    def walk(b):
        items = list(iter_klv(b))
        gps = next((x for x in items if x[0] in (b"GPS9", b"GPS5")), None)
        if gps:
            scal = next((x for x in items if x[0] == b"SCAL"), None)
            scales = struct.unpack(">9i", scal[4][:36]) if scal and len(scal[4]) >= 36 else (10000000, 10000000, 1000, 1000, 100, 1, 1000, 100, 1)
            _, _, sz, rep, payload = gps
            for r in range(rep):
                s = payload[r * sz : (r + 1) * sz]
                if len(s) < 32:
                    continue
                lat_i, lon_i, alt_i, spd2_i, _, days_i, secs_i, _, fix_i = struct.unpack(">7i2h", s[:32])
                if (fix_i / (scales[8] or 1)) < 2:
                    continue
                lat, lon = lat_i / scales[0], lon_i / scales[1]
                if abs(lat) < 0.01 and abs(lon) < 0.01:
                    continue
                t = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_i / (scales[5] or 1), seconds=secs_i / (scales[6] or 1))
                found.append({"lat": lat, "lon": lon, "alt": alt_i / (scales[2] or 1), "speed_kmh": (spd2_i / (scales[3] or 1)) * 3.6, "gps_time": t})
            return
        for key, typ, _, _, p in items:
            if typ == 0 and key in (b"DEVC", b"STRM", b"GPMF"):
                walk(p)
    walk(buf)
    return found


def load_gps_track(gpmd_path, crc_path, time_offset=0.0):
    if not (Path(gpmd_path).exists() and Path(crc_path).exists()):
        return []
    packets = []
    for line in Path(crc_path).read_text(errors="replace").splitlines():
        p = [x.strip() for x in line.split(",") if x.strip()]
        if len(p) >= 5 and not line.startswith("#"):
            try:
                packets.append((int(p[2]) / 1000.0, int(p[3]) / 1000.0, int(p[4])))
            except ValueError:
                pass
    blob, track, offset = Path(gpmd_path).read_bytes(), [], 0
    for i, (pts, dur, size) in enumerate(packets):
        chunk, offset = blob[offset : offset + size], offset + size
        if len(chunk) < size:
            break
        points = extract_gps_points(chunk)
        if not points:
            continue
        step = max(0.05, (packets[i + 1][0] if i + 1 < len(packets) else pts + max(dur, 0.1 * len(points))) - pts) / len(points)
        for s_idx, pt in enumerate(points):
            pt = dict(pt)
            pt["t"] = time_offset + pts + s_idx * step
            track.append(pt)
    track.sort(key=lambda x: x["t"])
    return track


def interpolate_gps(track, t):
    if not track:
        return None
    if t <= track[0]["t"]:
        return dict(track[0]) if track[0]["t"] - t <= 2 else None
    if t >= track[-1]["t"]:
        return dict(track[-1]) if t - track[-1]["t"] <= 2 else None
    lo, hi = 0, len(track) - 1
    while lo + 1 < hi:
        m = (lo + hi) // 2
        lo, hi = (m, hi) if track[m]["t"] <= t else (lo, m)
    p1, p2 = track[lo], track[hi]
    f = (t - p1["t"]) / (p2["t"] - p1["t"]) if p2["t"] > p1["t"] else 0.0
    return {
        "t": t, "lat": p1["lat"] + (p2["lat"] - p1["lat"]) * f, "lon": p1["lon"] + (p2["lon"] - p1["lon"]) * f,
        "alt": p1["alt"] + (p2["alt"] - p1["alt"]) * f, "speed_kmh": p1["speed_kmh"] + (p2["speed_kmh"] - p1["speed_kmh"]) * f,
        "gps_time": p1["gps_time"] + (p2["gps_time"] - p1["gps_time"]) * f,
    }


def chainage_at(track, cum, t, s_km, e_km, dur):
    if track and cum and cum[-1] > 1:
        if t <= track[0]["t"]:
            frac = 0.0
        elif t >= track[-1]["t"]:
            frac = 1.0
        else:
            lo, hi = 0, len(track) - 1
            while lo + 1 < hi:
                m = (lo + hi) // 2
                lo, hi = (m, hi) if track[m]["t"] <= t else (lo, m)
            span = track[hi]["t"] - track[lo]["t"]
            frac = (cum[lo] + (cum[hi] - cum[lo]) * ((t - track[lo]["t"]) / span if span > 0 else 0)) / cum[-1]
    elif dur > 0:
        frac = min(1.0, max(0.0, t / dur))
    else:
        return ""
    return format_chainage(s_km + frac * (e_km - s_km))


# ==============================================================================
# AWS S3 Upload & Video Frame Streamer
# ==============================================================================
def upload_to_s3(local_path, s3_key, bucket="rastaaibucket", region="ap-south-1"):
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    try:
        import boto3
        boto3.client("s3", region_name=region).upload_file(str(local_path), bucket, s3_key, ExtraArgs={"ContentType": "image/jpeg"})
        return url, True
    except Exception:
        try:
            res = subprocess.run(["aws", "s3", "cp", str(local_path), f"s3://{bucket}/{s3_key}"], capture_output=True, timeout=20)
            return url, res.returncode == 0
        except Exception:
            return url, False


def stream_video_frames(url, gpmd_path, crc_path, fps=3, seconds=None, seek=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if seek:
        cmd += ["-ss", str(seek)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += [
        "-i", url, "-map", "0:v:0", "-an", "-vf", f"fps={fps},scale=1920:1080:flags=bilinear,format=bgr24",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        "-map", "0:3?", "-c", "copy", "-f", "data", str(gpmd_path),
        "-map", "0:3?", "-c", "copy", "-f", "framecrc", str(crc_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_sz, idx = 1920 * 1080 * 3, 0
    while True:
        raw = proc.stdout.read(frame_sz)
        while len(raw) < frame_sz:
            chunk = proc.stdout.read(frame_sz - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) < frame_sz:
            break
        yield np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3), (float(seek) if seek else 0.0) + idx / fps
        idx += 1
    proc.stdout.close()
    proc.wait()


def draw_single_bump_box(img, bbox, conf):
    res = img.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    color = (0, 0, 255)  # Red box in BGR
    cv2.rectangle(res, (x1, y1), (x2, y2), color, 2)
    label = f"Speed-Bump {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(res, (x1, max(0, y1 - h - 8)), (x1 + w + 6, max(0, y1)), color, -1)
    cv2.putText(res, label, (x1 + 3, max(h + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    return res


def stamp_screenshot(img, row):
    res = img.copy()
    overlay = res.copy()
    cv2.rectangle(overlay, (12, 12), (540, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, res, 0.35, 0, res)
    lines = [
        f"GPS: {row['latitude']}, {row['longitude']}" if row["latitude"] else "GPS: Unavailable",
        f"Chainage: {row['estimated_chainage'] or row['start_chainage']}",
        f"Road: {row['road_no']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(res, line, (24, 42 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
    return res


def speed_bump_class_ids(names):
    """Return YOLO class indices that are exactly the speed-bump class (ignore all others)."""
    ids = []
    for idx, label in names.items():
        key = str(label).lower().replace("_", "-").replace(" ", "-")
        if key in ("speed-bump", "speedbump"):
            ids.append(int(idx))
    return ids


def is_plausible_speed_bump_bbox(bbox, frame_h=1080, frame_w=1920):
    """Reject wide/low boxes that match road texture false positives (see MDR-43 reference frames)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    cy = (y1 + y2) / 2.0
    max_h = frame_h * 0.075
    if h > max_h:
        return False
    if w < frame_w * 0.07:
        return False
    if cy > frame_h * 0.86:
        return False
    if (w / h) < 3.5:
        return False
    return True


def speed_bump_roi_texture_ok(frame, bbox, min_std=10.0):
    """Real speed tables show strong pavement contrast; flat seams/shadows stay below this threshold."""
    fh, fw = frame.shape[:2]
    x1 = max(0, min(fw - 1, int(round(bbox[0]))))
    y1 = max(0, min(fh - 1, int(round(bbox[1]))))
    x2 = max(x1 + 1, min(fw, int(round(bbox[2]))))
    y2 = max(y1 + 1, min(fh, int(round(bbox[3]))))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) >= min_std


def merge_events(events, meters=10.0, win=4.0):
    merged = []
    for e in sorted(events, key=lambda x: x["t"]):
        if not merged:
            merged.append(e)
            continue
        prev = merged[-1]
        same = (haversine_m(prev["lat"], prev["lon"], e["lat"], e["lon"]) <= meters) if (prev.get("lat") and e.get("lat")) else (e["t"] - prev["t"] <= win)
        if same:
            if e["conf"] > prev["conf"]:
                merged[-1] = e
        else:
            merged.append(e)
    return merged


# ==============================================================================
# Road Processing Core
# ==============================================================================
def process_road(model, bump_cls_ids, rec, out_dir, bucket, region, prefix, fps=3, batch_sz=8, conf=0.40, seconds=None, seek=None):
    road_no, road_name, sub = rec.get("roadNo", "Road"), rec.get("roadName", ""), rec.get("subDivision", "")
    s_km, e_km = parse_chainage(rec.get("startChainage")), parse_chainage(rec.get("endChainage"))
    work_dir = out_dir / "_work" / "".join(c if c.isalnum() else "_" for c in road_no)
    shot_dir = out_dir / "screenshots" / "".join(c if c.isalnum() else "_" for c in road_no)
    work_dir.mkdir(parents=True, exist_ok=True)
    shot_dir.mkdir(parents=True, exist_ok=True)

    urls = [rec.get(f"data_status[{i}].video_url") for i in (0, 1) if rec.get(f"data_status[{i}].video_url")]
    fns = [rec.get(f"data_status[{i}].fileName") or f"{road_no}_p{i+1}.mp4" for i in range(len(urls))]

    all_events, combined_track, tot_dur = [], [], 0.0
    for p_idx, (url, fn) in enumerate(zip(urls, fns), start=1):
        p_name, p_work = f"Part {p_idx}", work_dir / f"p{p_idx}"
        p_work.mkdir(parents=True, exist_ok=True)
        g_bin, g_crc = p_work / "g.bin", p_work / "g.crc"
        for p in (g_bin, g_crc):
            p.unlink(missing_ok=True)

        log(f"  Streaming [{p_name}] {fn}...")
        batch, b_times, cur, p_dur = [], [], None, 0.0

        def run_detection_batch(batch, b_times):
            nonlocal cur
            if not batch:
                return
            predict_kw = {"imgsz": 1280, "conf": conf, "verbose": False, "device": 0}
            if bump_cls_ids:
                predict_kw["classes"] = bump_cls_ids
            results = model.predict(batch, **predict_kw)
            fh, fw = batch[0].shape[:2]
            for f_img, t_glob, res in zip(batch, b_times, results):
                boxes = getattr(res, "boxes", None)
                bumps = []
                if boxes is not None and len(boxes) > 0:
                    for box, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                        bbox = [round(float(v), 1) for v in box]
                        if (
                            float(c) >= conf
                            and is_plausible_speed_bump_bbox(bbox, fh, fw)
                            and speed_bump_roi_texture_ok(f_img, bbox)
                        ):
                            bumps.append({"conf": float(c), "bbox": bbox})

                if not bumps:
                    if cur and t_glob - cur["last_t"] > 1.2:
                        all_events.append(cur)
                        cur = None
                    continue

                best = max(bumps, key=lambda x: x["conf"])
                merged_box = list(best["bbox"])
                for other in bumps:
                    if other is best:
                        continue
                    ob = other["bbox"]
                    y_over = max(0, min(merged_box[3], ob[3]) - max(merged_box[1], ob[1]))
                    h_min = min(merged_box[3] - merged_box[1], ob[3] - ob[1])
                    if h_min > 0 and (y_over / h_min) > 0.4:
                        merged_box[0] = min(merged_box[0], ob[0])
                        merged_box[1] = min(merged_box[1], ob[1])
                        merged_box[2] = max(merged_box[2], ob[2])
                        merged_box[3] = max(merged_box[3], ob[3])

                if not cur or t_glob - cur["last_t"] > 1.2:
                    if cur:
                        all_events.append(cur)
                    cur = {
                        "t": t_glob,
                        "last_t": t_glob,
                        "conf": best["conf"],
                        "bbox": merged_box,
                        "raw_image": f_img.copy(),
                        "part": p_name,
                        "file": fn,
                    }
                elif best["conf"] > cur["conf"]:
                    cur.update({
                        "t": t_glob,
                        "last_t": t_glob,
                        "conf": best["conf"],
                        "bbox": merged_box,
                        "raw_image": f_img.copy(),
                    })

        for frame, loc_t in stream_video_frames(url, g_bin, g_crc, fps=fps, seconds=seconds, seek=seek):
            p_dur = loc_t
            batch.append(frame)
            b_times.append(tot_dur + loc_t)

            if len(batch) >= batch_sz:
                run_detection_batch(batch, b_times)
                batch, b_times = [], []

        run_detection_batch(batch, b_times)

        if cur:
            all_events.append(cur)
        track = load_gps_track(g_bin, g_crc, time_offset=tot_dur)
        combined_track.extend(track)
        tot_dur += p_dur
        log(f"    [{p_name}] Finished ({p_dur:.1f}s), GPS samples: {len(track)}")

    combined_track.sort(key=lambda x: x["t"])
    cum = [0.0]
    for p1, p2 in zip(combined_track, combined_track[1:]):
        cum.append(cum[-1] + haversine_m(p1["lat"], p1["lon"], p2["lat"], p2["lon"]))

    located = []
    for e in all_events:
        gps = interpolate_gps(combined_track, e["t"])
        e = dict(e)
        e.update({"lat": gps["lat"], "lon": gps["lon"], "alt": gps["alt"], "speed_kmh": gps["speed_kmh"], "gps_time": gps["gps_time"]} if gps else {"lat": None, "lon": None, "alt": None, "speed_kmh": None, "gps_time": None})
        located.append(e)

    clean_pfx = prefix.strip("/")
    bumps = merge_events(located)
    det_rows, s3_urls = [], []
    for idx, b in enumerate(bumps, start=1):
        c_est = chainage_at(combined_track, cum, b["t"], s_km, e_km, tot_dur)
        lat_t = f"{b['lat']:.7f}" if b["lat"] else ""
        lon_t = f"{b['lon']:.7f}" if b["lon"] else ""
        shot_name = f"{Path(b['file']).stem}_bump{idx:02d}_t{b['t']:.1f}_{lat_t or 'nogps'}.jpg"
        local_path = shot_dir / shot_name
        s3_key = f"{clean_pfx}/{shot_dir.name}/{shot_name}"

        row = {
            "road_no": road_no, "road_name": road_name, "start_chainage": rec.get("startChainage", ""), "end_chainage": rec.get("endChainage", ""),
            "estimated_chainage": c_est, "sub_division": sub, "data_status[0].video_url": urls[0],
            "data_status[1].video_url": urls[1] if len(urls) > 1 else "", "data_status[0].fileName": fns[0],
            "data_status[1].fileName": fns[1] if len(fns) > 1 else "", "speed_bump_index": idx, "video_time_sec": f"{b['t']:.2f}",
            "source_video_part": b["part"], "gps_time_utc": b["gps_time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ") if b["gps_time"] else "",
            "latitude": lat_t, "longitude": lon_t, "altitude_m": f"{b['alt']:.1f}" if b["alt"] else "", "speed_kmh": f"{b['speed_kmh']:.1f}" if b["speed_kmh"] else "",
            "confidence": f"{b['conf']:.3f}", "bbox": " ".join(str(v) for v in b["bbox"]),
            "google_map_link": f"https://www.google.com/maps?q={lat_t},{lon_t}" if lat_t else "",
            "aws_s3_url": f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}",
            "local_screenshot": str(local_path.relative_to(out_dir)), "source_video_file": b["file"], "survey_date": rec.get("Date", ""),
        }
        annotated = draw_single_bump_box(b["raw_image"], b["bbox"], b["conf"])
        cv2.imwrite(str(local_path), stamp_screenshot(annotated, row), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        url, _ = upload_to_s3(local_path, s3_key, bucket=bucket, region=region)
        row["aws_s3_url"] = url
        s3_urls.append(url)
        det_rows.append(row)

    summary = {
        "road_no": road_no, "road_name": road_name, "start_chainage": rec.get("startChainage", ""), "end_chainage": rec.get("endChainage", ""),
        "sub_division": sub, "distance_km": rec.get("distance", ""), "data_status[0].video_url": urls[0],
        "data_status[1].video_url": urls[1] if len(urls) > 1 else "", "speed_bumps_count": len(bumps),
        "status": "PROCESSED", "all_aws_frame_urls": " ; ".join(s3_urls),
    }
    shutil.rmtree(work_dir, ignore_errors=True)
    return summary, det_rows


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Optimized Speed Bump Video Detector.")
    parser.add_argument("--sheet", default=str(DEFAULT_XLSX), help="Path to input Excel sheet")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to speedbump.pt")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to a new timestamped folder; existing results are never used)")
    parser.add_argument("--subdivision", default=None, help="Filter by sub-division")
    parser.add_argument("--bucket", default="rastaaibucket", help="AWS S3 bucket name")
    parser.add_argument("--region", default="ap-south-1", help="AWS region")
    parser.add_argument("--prefix", default=None, help="AWS S3 folder prefix (defaults to a unique run folder under SpeedBreaker/Div_Nilanga2/)")
    parser.add_argument("--fps", type=int, default=5, help="Sample FPS; higher sampling helps recover missed speed breakers")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold; lower threshold improves recall, review false positives")
    parser.add_argument("--seconds", type=float, default=None, help="Process first N seconds")
    parser.add_argument("--seek", type=float, default=None, help="Seek N seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit roads")
    parser.add_argument("--road-no", default=None, help="Process only this road number (e.g. MDR-43)")
    parser.add_argument("--force", action="store_true", help="Reprocess all roads")
    parser.add_argument("--sync-s3", action="store_true", help="Sync screenshots to S3")
    args = parser.parse_args()

    # Always isolate a new run by default. Never write into the previous output folder.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.prefix is None:
        args.prefix = f"SpeedBreaker/Div_Nilanga2/reprocessed_{run_stamp}"
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (DEFAULT_OUT / f"run_{run_stamp}").resolve()
    protected_outputs = ["speed_bumps_all_roads.csv", "road_summary.csv", "speed_bumps_all_roads.xlsx", "road_summary.xlsx"]
    if any((out_dir / name).exists() for name in protected_outputs):
        log(f"[Safety stop] Output reports already exist in {out_dir}. Choose a NEW --out-dir to protect prior results.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_s3:
        files = list((out_dir / "screenshots").rglob("*.jpg"))
        clean_pfx = args.prefix.strip("/")
        log(f"Syncing {len(files)} frames to AWS S3 bucket '{args.bucket}' under '{clean_pfx}/'...")
        for p in files:
            rel = p.relative_to(out_dir / "screenshots")
            upload_to_s3(p, f"{clean_pfx}/{rel}", bucket=args.bucket, region=args.region)
        log("S3 Sync complete.")
        return 0

    records = load_sheet(args.sheet, subdivision=args.subdivision)
    if args.road_no:
        want = args.road_no.strip()
        records = [r for r in records if r.get("roadNo", "").strip() == want]
    if args.limit:
        records = records[: args.limit]
    log(f"Loaded {len(records)} roads from {args.sheet}")
    if not records:
        log("[Error] No roads found. Check the input workbook and subdivision filter.")
        return 1

    from ultralytics import YOLO
    model = YOLO(args.model)
    bump_cls_ids = speed_bump_class_ids(model.names)
    if not bump_cls_ids:
        log("[Error] No Speed-Bump class found in model; check speedbump.pt class names.")
        return 1
    log(f"Using Speed-Bump class ids only: {bump_cls_ids} (ignoring all other model classes)")

    done_file = out_dir / "processed_roads.txt"
    done = set(done_file.read_text().splitlines()) if (done_file.exists() and not args.force and not args.seconds and not args.road_no) else set()

    det_csv, sum_csv = out_dir / "speed_bumps_all_roads.csv", out_dir / "road_summary.csv"

    def rewrite_csv_excluding_roads(path, fieldnames, exclude_roads):
        if not exclude_roads or not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if r.get("road_no", "").strip() not in exclude_roads]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in kept:
                w.writerow(r)

    if args.road_no and args.force:
        exclude = {args.road_no.strip()}
        rewrite_csv_excluding_roads(det_csv, DET_FIELDS, exclude)
        rewrite_csv_excluding_roads(sum_csv, SUM_FIELDS, exclude)
        shot_road = out_dir / "screenshots" / "".join(c if c.isalnum() else "_" for c in args.road_no.strip())
        if shot_road.exists():
            shutil.rmtree(shot_road, ignore_errors=True)
        if done_file.exists():
            lines = [ln for ln in done_file.read_text().splitlines() if ln.strip() != args.road_no.strip()]
            done_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        done.discard(args.road_no.strip())

    d_f = det_csv.open("a", newline="", encoding="utf-8")
    s_f = sum_csv.open("a", newline="", encoding="utf-8")
    d_w = csv.DictWriter(d_f, fieldnames=DET_FIELDS, extrasaction="ignore")
    s_w = csv.DictWriter(s_f, fieldnames=SUM_FIELDS, extrasaction="ignore")
    if not det_csv.exists() or det_csv.stat().st_size == 0:
        d_w.writeheader()
    if not sum_csv.exists() or sum_csv.stat().st_size == 0:
        s_w.writeheader()

    try:
        for idx, rec in enumerate(records, start=1):
            r_no = rec.get("roadNo", f"Road_{idx}")
            if r_no in done:
                log(f"[{idx}/{len(records)}] Skipping {r_no} (Already processed)")
                continue
            log(f"\n[{idx}/{len(records)}] Processing Road: {r_no} ({rec.get('roadName', '')})")
            try:
                summary, rows = process_road(model, bump_cls_ids, rec, out_dir, bucket=args.bucket, region=args.region, prefix=args.prefix, fps=args.fps, batch_sz=args.batch, conf=args.conf, seconds=args.seconds, seek=args.seek)
                for r in rows:
                    d_w.writerow(r)
                s_w.writerow(summary)
                d_f.flush()
                s_f.flush()
                if not args.seconds:
                    with done_file.open("a", encoding="utf-8") as dh:
                        dh.write(f"{r_no}\n")
            except Exception as e:
                log(f"[Error] Failed {r_no}: {e}")
    finally:
        d_f.close()
        s_f.close()

    # Generate Excel sheets from CSVs
    log("\nGenerating Excel workbooks...")
    with det_csv.open("r", encoding="utf-8") as f:
        r = list(csv.reader(f))
        if r:
            save_xlsx(out_dir / "speed_bumps_all_roads.xlsx", "Speed_Bumps", r[0], r[1:])
    with sum_csv.open("r", encoding="utf-8") as f:
        r = list(csv.reader(f))
        if r:
            save_xlsx(out_dir / "road_summary.xlsx", "Road_Summary", r[0], r[1:])
    log(f"Done! Reports saved in: {out_dir}")
    log("Review the new screenshots/reports for false positives and missed detections; lower thresholds can increase both recall and false alarms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
