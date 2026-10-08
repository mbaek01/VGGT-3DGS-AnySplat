# Local validation

Validated on 2026-10-07 using macOS arm64, Python 3.11.15, NumPy 1.26.4,
Pillow 10.4.0, PyYAML 6.0.2, scikit-image 0.24.0 and pytest 8.3.5.
GPU setup recipes target Linux x86_64 / Python 3.10 separately.

## Passed

- 15 CPU tests in `tests/test_core.py`.
- Exact kitchen and room assignment ranges, with 32/4 disjoint splits.
- Synthetic COLMAP binary parsing and 36-image shared preparation, including
  camera inversion, RGB separation and crop/intrinsic consistency.
- Known Sim(3) recovery and proper rotation/translation transformation.
- Degenerate center rejection and proof that modifying held-out camera centers
  cannot change the training-only fit.
- Synthetic four-view evaluator round trip, known PSNR, missing-frame and
  wrong-resolution failures. LPIPS explicitly disabled in these tests.
- CLI help without site packages for the orchestrator and both native workers.
- Data-free dry-run plan with exact train views and full resolved method settings.
- Explicit failure for the unimplemented `ours` adapter.
- Editable Python package installation and `dtbench` console entry point.
- YAML/JSON parsing, Python compilation, shell syntax and script help checks.
- Clean upstream source commits match the lockfile; real Git submodule entries
  preserve all three exact commit IDs.

## Not executed

- CUDA environment installation or CUDA kernel compilation.
- VGGT/AnySplat checkpoint loading and inference.
- gsplat optimization or either baseline's native target rendering.
- Actual Mip-NeRF 360 images/reference models.
- LPIPS pretrained-network execution or full benchmark scores/timing.

Native integration was inspected against the pinned source, including gsplat's
normalization and default split, VGGT's COLMAP export, and AnySplat's pose and
decoder conventions. That source inspection and the CPU tests do not replace a
Linux GPU integration run. Run the documented preflight commands and ten-step
smoke run before scheduling full 30,000-step experiments.
