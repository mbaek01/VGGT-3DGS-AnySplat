# Final method: RGB Pose+Focal → 3DGS

최종 Ours는 기존 **staged RGB Pose+Focal refinement**이다. Shared-K 또는 MPR을 사용하지 않는다. 기존 matcher, robust Sampson optimizer, SE(3) 계산, Gaussian initialization/training과 evaluator는 그대로 유지한다.

## Retained code

| 역할 | 파일 |
| --- | --- |
| 최종 실행 entrypoint | `benchmark/reproduce.py` |
| Training RGB SuperPoint/LightGlue + Fundamental MAGSAC | `benchmark/rgb_pose_refinement/matching.py` |
| Sampson geometry / pose prior | `benchmark/rgb_pose_refinement/geometry.py`, `optimizer.py` |
| Shared focal + pose optimizer | `benchmark/intrinsic_refinement/optimizer.py` |
| Camera/right SE(3), normalized translation | `benchmark/pose_refinement/se3.py` |
| Frozen train cameras + native GS | `benchmark/workers/pose_focal.py` |
| Native config / initialization / RNG checks | `benchmark/utils/native.py` |
| VGGT export / native GS / common evaluator | `benchmark/methods/vggt_3dgs.py`, `benchmark/evaluation/` |
| 비교용 AnySplat baseline | `benchmark/methods/anysplat.py`, `configs/methods/anysplat.yaml` |

MPR, Shared-K, Oracle, focal sweep, synthetic recovery experiment drivers, attribution queues, posthoc audit scripts와 관련 config/test는 제거했다. Optimizer 수학과 matching source는 변경하지 않았다. 기존 `outputs/`, `data/`, `work/prepared/`, checkpoint와 보고서의 numeric artifacts는 보존한다. 이전 보고서/로그의 삭제된 module 경로는 당시 실행 이력이다.

## Environment and baseline source

Root [README](../README.md)의 설치와 data preparation 명령을 사용한다. Ours에는 `scripts/setup.sh submodules`, `eval`, `vggt`, `gsplat`이 필요하며, AnySplat baseline까지 실행할 때는 `anysplat`도 설치한다. `scripts/setup.sh all`로 모두 설치할 수도 있다. 설치 script에는 pinned LightGlue도 이미 포함되어 있다. Matching은 `dtb-vggt`, native Parser/GS는 `dtb-gsplat`, scoring은 `dtb-eval` 환경을 사용한다. 최종 entrypoint는 `dtb-gsplat`에서 실행한다. 모델 가중치는 최초 실행 시 자동 다운로드하거나 `scripts/prefetch.py`로 미리 다운로드할 수 있다.

`.gitignore`는 최종 pipeline, 두 baseline과 테스트에 필요한 파일을 포함한다. `configs/methods/rgb_pose_refine.yaml`은 최종 설정에서 직접 읽으며, pose-only optimizer의 공통 함수와 `pose_refinement/se3.py`도 사용하므로 유지한다. AnySplat adapter/config/environment와 baseline 실행·모델 다운로드 script도 포함한다. `third_party/`, `.gitmodules`, `third_party.lock.json`은 현재 `check_pins()`가 모든 upstream을 검사하므로 유지한다. MPR·Shared-K·Oracle와 중간 실험 driver는 배포 대상에서 제외한다.

새 workspace에서는 먼저 VGGT baseline을 만든다. 이미 완료된 baseline이 있으면 재사용한다.

```bash
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene kitchen --method vggt_3dgs --run-id final-source
```

Baseline source에는 `train/images`, VGGT `train/sparse`, `native/cfg.yml`, `predicted_train_cameras.json`, `target_cameras.json`, environment/provenance 및 완료된 job metadata가 필요하다. Source baseline의 GS budget과 seed42를 그대로 따른다. Report의 최종 score는 30k source이다.

## Reproduce from the baseline export

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n dtb-gsplat python -m benchmark.reproduce \
  --source-run outputs/vggt_3dgs/kitchen/final-source \
  --output-dir outputs/ours/kitchen/final-reproduction-v1
```

Train RGB matching → CPU float64 staged refinement → unchanged native GS → common CPU evaluation을 실행한다. Room도 동일하게 scene/source/output 경로만 바꾼다. 새 output directory만 허용한다. Matching에는 최초 실행 시 pinned SuperPoint/LightGlue weights가 필요하며 기존 cache를 우선 사용한다.

Plan만 확인하려면 같은 명령에 `--dry-run`을 붙인다. `--stage refine`은 camera 결과만 만들고 Gaussian 학습을 하지 않는다. 이후 동일 source/output에 `--stage train`, `--stage evaluate`를 순서대로 실행할 수 있다. `train`은 이미 Gaussian checkpoint가 있으면 덮어쓰지 않고 실패한다. `evaluate`는 지정한 새 재현 run의 metrics를 공통 evaluator로 계산한다.

## Reuse original cached RGB matches

아래는 이 workspace의 기존 matches를 이용해 **동일 optimizer를 다시 실행**하는 예이다. Cache의 matching code hash, settings, image order와 train RGB hashes가 일치해야 한다.

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n dtb-gsplat python -m benchmark.reproduce \
  --source-run outputs/vggt_3dgs/room/benchmark-v1 \
  --match-cache outputs/rgb_pose_refine/rgb-correspondence-20261010-v2/room/rgb_matches \
  --output-dir outputs/ours/room/cached-matches-v1
```

`--stage refine`을 추가하면 GS 없이 원래 camera refinement의 수치 재현만 확인한다.

## Reuse final cameras and/or Gaussian checkpoint

기존 staged camera 결과로 GS를 재학습하려면 `--refinement`를 지정한다. 이 경우 matching과 camera optimization을 생략한다. `result.npz` 옆의 `settings.json`과 최종 config가 정확히 일치해야 한다.

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n dtb-gsplat python -m benchmark.reproduce \
  --source-run outputs/vggt_3dgs/room/benchmark-v1 \
  --refinement outputs/rgb_pose_focal_refine/pose-focal-20261010-v1/room/pose_focal_staged/result.npz \
  --output-dir outputs/ours/room/saved-cameras-v1
```

Gaussian training 없이 저장된 최종 checkpoint를 다시 렌더링하려면 위 명령에 다음을 추가한다. 원래 checkpoint와 refinement의 짝을 유지한다.

```bash
--render-checkpoint outputs/focal_nvs_validation/nvs-focal-20261010-v1/room/pose_focal_30000/native/final.pt
```

Kitchen의 saved refinement는 `outputs/rgb_pose_focal_refine/pose-focal-20261010-v1/kitchen/pose_focal_staged/result.npz`, final Gaussian checkpoint는 `outputs/focal_attribution/controlled-20261010-v1/kitchen/pose_focal_30000_seed42/native/final.pt`이다. 삭제된 실험 package 이름이 output 경로에 남아 있어도 해당 파일은 그대로 재사용할 수 있다.

## Frozen protocol and settings

- Kitchen/Room 각각 기존 32 train / 4 test split, 448×448 prepared RGB를 유지한다.
- Pose는 `T_i = T_i0 @ exp(delta_i)`이며 첫 camera correction을 고정한다.
- Scene당 scalar `s=exp(a)`로 train fx/fy만 수정한다. Principal point와 fx/fy ratio를 유지하며 bound는 ±10%이다.
- Staged schedule은 focal-only200 + joint400 update이다. Reference pose/K와 test RGB는 optimizer 입력이 아니다.
- GS 초기화 뒤 train poses/K만 설치한다. VGGT points, parser normalization, initialization policy, optimizer, density control과 RNG는 유지한다.
- Test c2w와 Sim(3)는 원래 VGGT train centers 기준으로 고정한다. **모든 방법의 primary test K는 같은 prepared reference K**이며 learned focal scale을 test K에 곱하지 않는다.
- PSNR/SSIM/Alex LPIPS는 기존 공통 evaluator를 사용한다. 공식 과제의 test-K 규정은 확인되지 않았다.

## Verification and report

```bash
bash scripts/test.sh
```

기존 geometry/matching/optimizer tests와 최종 reproduction의 cache, saved-camera installation, config/RNG invariants를 검사한다. 저장된 refinement 재사용은 최적화를 생략한다.

연구 보고서, 성능 CSV와 평가 audit는 로컬 `reports/`에 보존하며 GitHub 배포에서 제외한다. Reconstruction의 NVS metric을 physical 3D geometry accuracy로 확대 해석하지 않는다.
