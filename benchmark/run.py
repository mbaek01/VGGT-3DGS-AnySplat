"""CPU orchestration CLI. --help imports no third-party numerical/native packages."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from benchmark.utils.io import ROOT, read_json, read_yaml, write_json, root_path, sha256


def configuration(args):
    cfg = read_yaml(root_path(args.config))
    if getattr(args, "data_root", None):
        cfg["data_root"] = str(Path(args.data_root).expanduser().resolve())
    if getattr(args, "alignment", None):
        cfg["alignment"]["mode"] = args.alignment
    return cfg


def scene_config(scene):
    return read_yaml(ROOT / "configs/scenes" / f"{scene}.yaml")


def check_pins():
    lock = read_json(ROOT / "third_party.lock.json")
    for name, pin in lock.items():
        folder = ROOT / "third_party" / name
        if not (folder / ".git").exists():
            raise RuntimeError(f"Missing {name}: run bash scripts/setup.sh submodules")
        head = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if head != pin["commit"]:
            raise RuntimeError(f"{name} at {head}, expected {pin['commit']}")
        dirty = subprocess.check_output(["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty:
            raise RuntimeError(f"Tracked upstream modifications in {name}; restore or explicitly update the lock")
    return lock


def validate():
    from benchmark.datasets.mipnerf360 import load_split
    for scene in ("kitchen", "room"):
        split = load_split(ROOT / "splits" / f"{scene}.json")
        print(f"{scene}: {len(split['train'])} train, {len(split['test'])} test")
    pins = check_pins()
    for name, pin in pins.items():
        print(f"{name}: {pin['commit']}")


def prepare_one(cfg, scene):
    from benchmark.datasets.mipnerf360 import prepare
    sc = scene_config(scene)
    dest = root_path(cfg["prepared_root"]) / scene
    prepare(root_path(cfg["data_root"]) / scene, sc, root_path(sc["split"]), dest, cfg["preprocessing"])
    print(dest)
    return dest


def execute_stage(command, log, cwd):
    start = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Running {log.stem}; log: {log}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter()-start


def run_one(args, cfg, scene, method):
    from benchmark.datasets.mipnerf360 import load_split
    method_cfg = read_yaml(ROOT / "configs/methods" / f"{method}.yaml")
    if method == "ours":
        raise NotImplementedError("'ours' is a placeholder, not an executable baseline")
    if method == "anysplat" and cfg["preprocessing"]["width"] != cfg["preprocessing"]["height"]:
        raise ValueError("Use square shared crops: native AnySplat preprocessing otherwise changes the field of view")
    if args.iterations is not None:
        if args.iterations < 1:
            raise ValueError("iterations must be positive")
        if method == "vggt_3dgs":
            method_cfg["gsplat"]["iterations"] = args.iterations
    split = load_split(root_path(scene_config(scene)["split"]))
    if args.dry_run:
        print(json.dumps({"scene": scene, "method": method, "split": split,
             "alignment": cfg["alignment"], "preprocessing": cfg["preprocessing"],
             "method_config": method_cfg, "environments": cfg["environments"]}, indent=2))
        return
    pins = check_pins()
    prepared = root_path(cfg["prepared_root"]) / scene
    if not prepared.exists():
        prepare_one(cfg, scene)
    meta = read_json(prepared / "cameras.json")
    if meta["split"] != split or meta["preprocessing"] != cfg["preprocessing"]:
        raise ValueError("Prepared data has a different split/preprocessing; use a new prepared_root")
    provenance = read_json(prepared / "provenance.json")
    for name in split["train"]:
        if sha256(prepared / "train/images" / f"{name}.png") != provenance["images"][name]["prepared_sha256"]:
            raise ValueError(f"Training image changed: {name}")
    if {p.stem for p in (prepared / "train/images").iterdir()} != set(split["train"]):
        raise ValueError("Train directory must contain exactly the 32 training images")
    if method == "anysplat" and method_cfg["input_size"] != 448:
        raise ValueError("Pinned AnySplat process_image requires input_size=448")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    out = root_path(cfg["output_root"]) / method / scene / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    train_dir = out / "train"
    shutil.copytree(prepared / "train", train_dir)
    job = {"schema_version": 1, "method": method, "method_config": method_cfg,
           "metadata": meta, "alignment": cfg["alignment"], "pins": pins,
           "run_dir": str(out), "train_dir": str(train_dir)}
    write_json(out / "job.json", job)
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "dataset_provenance.json", provenance)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "status.json", status)
    timing = {"stage_wall_seconds": {}, "definitions": {
        "stage_wall_seconds": "Each subprocess incl. startup, model load/download, native work and file I/O; excludes shared preparation and metrics.",
        "reconstruction_compute_seconds": "VGGT forward + gsplat native training loop, or AnySplat inference. Training includes its native logging/checkpoint I/O. VGGT BA/export excluded; see vggt.native_export_seconds.",
        "render_seconds": "CUDA synchronized native target rendering, excludes PNG encoding and transfer to CPU."}}
    try:
        if method == "vggt_3dgs":
            command = cfg["environments"]["vggt"] + ["-m", "benchmark.workers.vggt_export", "--job", str(out / "job.json")]
            timing["stage_wall_seconds"]["vggt"] = execute_stage(command, out / "logs/vggt.log", ROOT / "third_party/vggt")
            timing["vggt"] = read_json(out / "vggt_timing.json")
        environment = "gsplat" if method == "vggt_3dgs" else "anysplat"
        command = cfg["environments"][environment] + ["-m", "benchmark.workers.native", "--job", str(out / "job.json")]
        timing["stage_wall_seconds"][environment] = execute_stage(command, out / f"logs/{environment}.log", ROOT / "third_party" / environment)
        timing["native"] = read_json(out / "native_timing.json")
        timing["reconstruction_compute_seconds"] = (timing["vggt"]["inference_seconds"] + timing["native"]["optimization_seconds"] if method == "vggt_3dgs" else timing["native"]["inference_seconds"])
        timing["render_seconds"] = timing["native"]["render_seconds"]
        write_json(out / "timing.json", timing)
        if not args.skip_evaluation:
            from benchmark.evaluation.evaluator import evaluate
            evaluate(out, prepared, cfg["evaluation"], args.skip_lpips)
        status["status"] = "rendered" if args.skip_evaluation else ("complete_partial_metrics" if args.skip_lpips else "complete")
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "status.json", status)
    print(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed 32/4-view Mip-NeRF 360 benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Global YAML; relative paths resolve from repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate assignment splits and exact clean upstream pins")
    prep = sub.add_parser("prepare", help="Prepare shared train-only images, reference cameras and evaluator-only GT")
    prep.add_argument("--scene", choices=["kitchen", "room", "all"], default="all")
    prep.add_argument("--data-root", type=Path)
    run = sub.add_parser("run", help="Reconstruct, render four test cameras and evaluate")
    run.add_argument("--scene", choices=["kitchen", "room", "all"], default="all")
    run.add_argument("--method", choices=["vggt_3dgs", "anysplat", "ours", "all"], default="all")
    run.add_argument("--data-root", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--iterations", type=int, help="Override gsplat steps (smoke tests are not benchmark results)")
    run.add_argument("--alignment", choices=["sim3_train_centers", "se3_train_centers", "none"])
    run.add_argument("--dry-run", action="store_true", help="Print plans without data, CUDA, downloads or mutations")
    run.add_argument("--skip-evaluation", action="store_true")
    run.add_argument("--skip-lpips", action="store_true", help="Development-only incomplete metrics; LPIPS is recorded as null")
    ev = sub.add_parser("evaluate", help="Evaluate an existing run with the common evaluator")
    ev.add_argument("--run-dir", type=Path, required=True)
    ev.add_argument("--prepared-dir", type=Path, required=True)
    ev.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = configuration(args)
        if args.command == "check":
            validate()
        elif args.command == "prepare":
            for scene in cfg["scenes"] if args.scene == "all" else [args.scene]:
                prepare_one(cfg, scene)
        elif args.command == "run":
            for method in cfg["methods"] if args.method == "all" else [args.method]:
                for scene in cfg["scenes"] if args.scene == "all" else [args.scene]:
                    run_one(args, cfg, scene, method)
        else:
            from benchmark.evaluation.evaluator import evaluate
            result = evaluate(args.run_dir.resolve(), args.prepared_dir.resolve(), cfg["evaluation"], args.skip_lpips)
            print(json.dumps(result["mean"], indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, NotImplementedError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
