import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .base import Baseline
from .common import target_cameras, save_rgb
from benchmark.utils.io import ROOT
from benchmark.utils.timing import cuda_timer, hardware


class AnySplatBaseline(Baseline):
    def reconstruct(self, job):
        import torch
        sys.path.insert(0, str(ROOT / "third_party/anysplat"))
        from src.model.model.anysplat import AnySplat
        from src.model.encoder.vggt.models.vggt import VGGT
        from src.utils.image import process_image
        cfg = job["method_config"]
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        # Upstream initializes its encoder from VGGT before loading AnySplat.
        # Scope a revision-only hook to that constructor; do not edit upstream.
        native_load = VGGT.from_pretrained
        def pinned_vggt(repo, **kwargs):
            if repo == "facebook/VGGT-1B":
                kwargs["revision"] = cfg["vggt_checkpoint_revision"]
            return native_load(repo, **kwargs)
        with patch.object(VGGT, "from_pretrained", side_effect=pinned_vggt):
            model = AnySplat.from_pretrained(cfg["checkpoint_repo"], revision=cfg["checkpoint_revision"]).cuda().eval()
        model.requires_grad_(False)
        names = job["metadata"]["split"]["train"]
        images = torch.stack([process_image(str(Path(job["train_dir"]) / "images" / f"{n}.png")) for n in names])[None].cuda()
        images = (images+1)*0.5
        timing = {}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), cuda_timer(timing, "inference_seconds"):
            gaussians, poses = model.inference(images)
        predicted = dict(zip(names, poses["extrinsic"][0].float().cpu().numpy()))
        out = Path(job["run_dir"]) / "native"
        out.mkdir(exist_ok=True)
        # Tensor-only native Gaussian fields; no interchange with the gsplat baseline.
        torch.save({"gaussians": {k: v.detach().cpu() for k, v in vars(gaussians).items()},
                    "poses": {k: v.detach().cpu() for k, v in poses.items()}}, out / "reconstruction.pt")
        timing["hardware"] = hardware()
        timing["num_gaussians"] = int(gaussians.means.shape[1])
        return {"model": model, "gaussians": gaussians, "predicted": predicted, "timing": timing}

    def render_test_views(self, job, reconstruction):
        import torch
        targets = target_cameras(job, reconstruction["predicted"])
        names = list(targets)
        cams = [job["metadata"]["cameras"][n] for n in names]
        width, height = cams[0]["width"], cams[0]["height"]
        poses = torch.tensor(np.stack(list(targets.values())), dtype=torch.float32, device="cuda")[None]
        K = torch.tensor(np.stack([c["K"] for c in cams]), dtype=torch.float32, device="cuda")[None]
        K[..., 0, :] /= width
        K[..., 1, :] /= height
        cfg = job["method_config"]
        near = torch.full((1, len(names)), cfg["near"], device="cuda")
        far = torch.full((1, len(names)), cfg["far"], device="cuda")
        timing = reconstruction["timing"]
        with torch.no_grad(), cuda_timer(timing, "render_seconds"):
            output = reconstruction["model"].decoder.forward(
                reconstruction["gaussians"], poses, K, near, far, (height, width))
        for name, rgb in zip(names, output.color[0]):
            save_rgb(job["run_dir"], name, rgb.permute(1, 2, 0).float().cpu().numpy())
        return timing
