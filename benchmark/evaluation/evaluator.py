from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.utils.io import read_json, write_json, sha256
from benchmark.evaluation.metrics import Metrics


def evaluate(run_dir, prepared_dir, cfg, skip_lpips=False):
    run_dir, prepared_dir = Path(run_dir), Path(prepared_dir)
    job = read_json(run_dir / "job.json")
    meta = read_json(prepared_dir / "cameras.json")
    if meta != job["metadata"]:
        raise ValueError("Prepared metadata does not match this run")
    expected = set(meta["split"]["test"])
    found = {p.stem for p in (run_dir / "renders").glob("*.png")}
    if found != expected:
        raise ValueError(f"Expected exactly {sorted(expected)} renders; found {sorted(found)}")
    metric = Metrics(cfg["lpips_net"], cfg["device"], skip_lpips)
    provenance = read_json(prepared_dir / "provenance.json")
    views = {}
    for name in meta["split"]["test"]:
        path = prepared_dir / "gt" / f"{name}.png"
        if sha256(path) != provenance["images"][name]["prepared_sha256"]:
            raise ValueError(f"Ground truth changed: {name}")
        with Image.open(run_dir / "renders" / f"{name}.png") as p, Image.open(path) as g:
            if p.mode != "RGB" or g.mode != "RGB" or p.size != g.size:
                raise ValueError(f"RGB/resolution mismatch for {name}; no implicit resize is allowed")
            views[name] = metric(np.asarray(p, dtype=np.float32)/255, np.asarray(g, dtype=np.float32)/255)
    means = {k: (float(np.mean([v[k] for v in views.values()])) if all(v[k] is not None for v in views.values()) else None) for k in ("psnr", "ssim", "lpips")}
    result = {"scene": meta["scene"], "method": job["method"], "views": views, "mean": means,
              "complete_metrics": not skip_lpips, "protocol": cfg,
              "alignment": job["alignment"], "preprocessing": meta["preprocessing"],
              "timing": read_json(run_dir / "timing.json")}
    write_json(run_dir / "metrics.json", result)
    return result
