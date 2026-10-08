#!/usr/bin/env python3
"""Download pinned model files before timing (run in dtb-vggt or dtb-anysplat)."""
import argparse
from benchmark.utils.io import ROOT, read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", "vggt", "anysplat"], default="all")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    vggt = read_yaml(ROOT / "configs/methods/vggt_3dgs.yaml")["vggt"]
    anysplat = read_yaml(ROOT / "configs/methods/anysplat.yaml")
    # AnySplat also initializes a VGGT encoder before loading its own checkpoint.
    configs = [vggt] if args.method == "vggt" else [vggt, anysplat]
    for cfg in configs:
        print(snapshot_download(cfg["checkpoint_repo"], revision=cfg["checkpoint_revision"],
            allow_patterns=["config.json", "model.pt", "model.safetensors", "pytorch_model.bin"]))


if __name__ == "__main__":
    main()
