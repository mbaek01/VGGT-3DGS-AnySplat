import argparse
import importlib.metadata
from pathlib import Path
from benchmark.utils.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Run a baseline's native reconstruction and rendering")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = read_json(args.job)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Native baselines require a Linux NVIDIA CUDA environment")
    from benchmark.methods import get_method
    method = get_method(job["method"])
    state = method.reconstruct(job)
    timing = method.render_test_views(job, state)
    write_json(Path(job["run_dir"]) / "native_timing.json", timing)
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(Path(job["run_dir"]) / "native_environment.json", versions)


if __name__ == "__main__":
    main()
