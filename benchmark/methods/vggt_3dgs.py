"""Small integration around the pinned gsplat native Runner; no upstream edits."""
import sys
from pathlib import Path
import numpy as np
from .base import Baseline
from .common import target_cameras, save_rgb
from benchmark.utils.io import ROOT, write_json
from benchmark.utils.timing import cuda_timer, hardware


class VGGT3DGS(Baseline):
    def reconstruct(self, job):
        import torch
        sys.path.insert(0, str(ROOT / "third_party/gsplat/examples"))
        from simple_trainer import Config, Runner
        settings = job["method_config"]["gsplat"]
        if job["method_config"]["seed"] != 42:
            raise ValueError("The pinned native gsplat Runner seeds to 42; this adapter requires seed=42")
        cfg = Config(data_dir=job["train_dir"], data_factor=1,
                     result_dir=str(Path(job["run_dir"]) / "native"), disable_viewer=True,
                     max_steps=settings["iterations"], sh_degree=settings["sh_degree"],
                     packed=settings["packed"], antialiased=settings["antialiased"],
                     near_plane=settings["near_plane"], far_plane=settings["far_plane"],
                     eval_steps=[], save_steps=[settings["iterations"]],
                     pose_opt=False, app_opt=False, test_every=8)
        runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        names = [Path(n).stem for n in runner.parser.image_names]
        if set(names) != set(job["metadata"]["split"]["train"]) or len(names) != 32:
            raise ValueError("VGGT export must contain exactly all 32 training cameras")
        # The native Dataset stores its selection in indices. Preserve its pixel
        # loading, losses, optimizer and densification; select ALL supplied views.
        runner.trainset.indices = np.arange(len(names))
        runner.valset.indices = np.array([], dtype=int)
        write_json(Path(job["run_dir"]) / "training_views.json", names)
        timing = {}
        torch.cuda.reset_peak_memory_stats()
        with cuda_timer(timing, "optimization_seconds"):
            runner.train()
        # The native trainer saves before the last update. Save actual final state
        # explicitly, so these renders and this checkpoint refer to the same state.
        torch.save({"splats": runner.splats.state_dict(), "completed_steps": cfg.max_steps},
                   Path(job["run_dir"]) / "native/final.pt")
        predicted = dict(zip(names, runner.parser.camtoworlds))
        timing["hardware"] = hardware()
        timing["num_gaussians"] = len(runner.splats["means"])
        return {"runner": runner, "predicted": predicted, "timing": timing}

    def render_test_views(self, job, reconstruction):
        import torch
        runner = reconstruction["runner"]
        targets = target_cameras(job, reconstruction["predicted"])
        timing, arrays = reconstruction["timing"], {}
        # Match the SH degree reached by the native schedule, including short smoke runs.
        degree = min((runner.cfg.max_steps-1)//runner.cfg.sh_degree_interval, runner.cfg.sh_degree)
        with torch.no_grad(), cuda_timer(timing, "render_seconds"):
            for name, pose in targets.items():
                cam = job["metadata"]["cameras"][name]
                colors, _, _ = runner.rasterize_splats(
                    camtoworlds=torch.tensor(pose, dtype=torch.float32, device=runner.device)[None],
                    Ks=torch.tensor(cam["K"], dtype=torch.float32, device=runner.device)[None],
                    width=cam["width"], height=cam["height"], sh_degree=degree,
                    near_plane=runner.cfg.near_plane, far_plane=runner.cfg.far_plane)
                arrays[name] = colors[0, ..., :3]
        for name, colors in arrays.items():
            save_rgb(job["run_dir"], name, colors.cpu().numpy())
        runner.writer.close()
        return timing
