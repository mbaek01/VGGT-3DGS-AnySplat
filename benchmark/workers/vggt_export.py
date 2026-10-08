"""Call upstream demo_fn, adding pinned weights, stable ordering and timing hooks."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
from benchmark.utils.io import ROOT, read_json, write_json
from benchmark.utils.timing import cuda_timer, hardware


def main():
    parser = argparse.ArgumentParser(description="VGGT native COLMAP export of 32 train views")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    import torch
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT native COLMAP demo requires CUDA")
    sys.path.insert(0, str(ROOT / "third_party/vggt"))
    import demo_colmap as demo
    cfg = job["method_config"]["vggt"]
    path = hf_hub_download(cfg["checkpoint_repo"], "model.pt", revision=cfg["checkpoint_revision"])
    weights = torch.load(path, map_location="cpu", weights_only=True)
    original_load = torch.hub.load_state_dict_from_url
    original_forward = demo.run_VGGT
    original_glob = demo.glob
    timing = {}

    def pinned_load(url, *args, **kwargs):
        if url == "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt":
            return weights
        return original_load(url, *args, **kwargs)

    def timed_forward(*args, **kwargs):
        with cuda_timer(timing, "inference_seconds"):
            return original_forward(*args, **kwargs)

    demo.glob = SimpleNamespace(glob=lambda pattern: sorted(original_glob.glob(pattern)))
    demo.run_VGGT = timed_forward
    torch.hub.load_state_dict_from_url = pinned_load
    native_args = SimpleNamespace(scene_dir=job["train_dir"], seed=job["method_config"]["seed"],
        use_ba=cfg["use_ba"], max_reproj_error=8., shared_camera=False,
        camera_type="SIMPLE_PINHOLE", vis_thresh=0.2, query_frame_num=8,
        max_query_pts=4096, fine_tracking=True, conf_thres_value=cfg["conf_thres_value"])
    try:
        torch.cuda.reset_peak_memory_stats()
        with cuda_timer(timing, "native_export_seconds"):
            demo.demo_fn(native_args)
    finally:
        torch.hub.load_state_dict_from_url = original_load
        demo.run_VGGT = original_forward
        demo.glob = original_glob
    timing["hardware"] = hardware()
    write_json(Path(job["run_dir"]) / "vggt_timing.json", timing)
    write_json(Path(job["run_dir"]) / "vggt_environment.json",
               {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})


if __name__ == "__main__":
    main()
