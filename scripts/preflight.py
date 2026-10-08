#!/usr/bin/env python3
"""Check native imports and installed rasterizer before expensive runs."""
import argparse
import sys
from benchmark.utils.io import ROOT
from benchmark.run import check_pins

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("method", choices=["vggt", "gsplat", "anysplat"])
args = parser.parse_args()
check_pins()
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this environment")
if args.method == "vggt":
    sys.path.insert(0, str(ROOT / "third_party/vggt"))
    import demo_colmap
elif args.method == "gsplat":
    sys.path.insert(0, str(ROOT / "third_party/gsplat/examples"))
    import simple_trainer
else:
    sys.path.insert(0, str(ROOT / "third_party/anysplat"))
    from src.model.model.anysplat import AnySplat
if args.method != "vggt":
    import gsplat
    expected = "1.3.0" if args.method == "gsplat" else "1.4.0"
    if not gsplat.__version__.startswith(expected):
        raise SystemExit(f"Expected gsplat {expected}, found {gsplat.__version__}")
print(f"{args.method}: imports OK; GPU={torch.cuda.get_device_name()}; torch={torch.__version__}")
