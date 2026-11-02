#!/usr/bin/env python3
"""Detect speed bumps across all survey videos in the road sheet.

This script processes survey videos, combines multi-part GoPro videos,
runs YOLO inference with speedbump.pt, groups detections, stamps annotated
frames, uploads to AWS S3, and generates consolidated Excel and CSV reports.

Forwards to process_all_roads_speedbumps.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from process_all_roads_speedbumps import main

if __name__ == "__main__":
    sys.exit(main())
