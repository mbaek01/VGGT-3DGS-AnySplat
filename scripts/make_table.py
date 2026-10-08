#!/usr/bin/env python3
"""Print one row per completed evaluation; never mix run IDs into an average."""
import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.outputs.glob("*/*/*/metrics.json")):
        data = json.loads(path.read_text())
        rows.append({"method": data["method"], "scene": data["scene"], "run": path.parent.name,
          "psnr": data["mean"]["psnr"], "ssim": data["mean"]["ssim"], "lpips": data["mean"]["lpips"],
          "reconstruction_s": data["timing"]["reconstruction_compute_seconds"],
          "render_s": data["timing"]["render_seconds"], "complete_metrics": data["complete_metrics"],
          "alignment": data["alignment"]["mode"], "resolution": f"{data['preprocessing']['width']}x{data['preprocessing']['height']}"})
    fields = ["method", "scene", "run", "psnr", "ssim", "lpips", "reconstruction_s", "render_s", "complete_metrics", "alignment", "resolution"]
    stream = args.csv.open("w", newline="") if args.csv else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.csv:
            stream.close()


if __name__ == "__main__":
    main()
