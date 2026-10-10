# Digital Twin Benchmark

**VGGT+3DGS**, **AnySplat**, 최종 Ours인 **staged RGB Pose+Focal refinement → VGGT points 기반 3DGS**를 실행하는 저장소입니다. GitHub에는 세 방법의 실행 코드, 의존성, 설정과 검증 테스트를 포함합니다. Ours의 상세 실행 방법은 [최종 방법 재현 문서](docs/REPRODUCE_OURS.md)를 참고하세요.

## 고정 평가 조건

| 장면 | 학습 / 테스트 | 테스트 이미지 |
| --- | --- | --- |
| Kitchen | 32 / 4 | DSCF0664, DSCF0673, DSCF0682, DSCF0691 |
| Room | 32 / 4 | DSCF4675, DSCF4684, DSCF4693, DSCF4702 |

- Training RGB에만 SuperPoint/LightGlue matching과 camera refinement를 적용합니다.
- Focal-only 200회 후 pose와 shared focal scale을 400회 공동 최적화합니다. Principal point는 고정합니다.
- VGGT points, Gaussian 초기화, scene normalization, seed42와 native GS 설정을 source baseline과 동일하게 유지합니다.
- 전처리와 렌더링은 448×448입니다. Primary test K는 prepared reference COLMAP K이며 learned focal scale을 곱하지 않습니다.
- 각 baseline의 predicted train cameras로 Sim(3)를 계산해 reference test pose를 native 좌표계로 변환합니다. Ours는 원래 VGGT baseline의 정렬을 고정합니다. PSNR/SSIM/Alex LPIPS는 공통 evaluator로 계산합니다.
- Reference pose/K와 held-out RGB는 refinement에 사용하지 않습니다. 공식 과제의 test-K 규정은 확인되지 않았습니다.

## 설치와 데이터 준비

Linux x86_64, conda, NVIDIA GPU/드라이버, CUDA 12.1 toolkit(`nvcc`)과 C++ compiler가 필요합니다. 아래 명령은 두 baseline과 Ours에 필요한 환경을 설치합니다.

```bash
export PYTHONNOUSERSITE=1
bash scripts/setup.sh submodules
bash scripts/setup.sh eval
bash scripts/setup.sh vggt
bash scripts/setup.sh gsplat
bash scripts/setup.sh anysplat

conda run --no-capture-output -n dtb-eval python -m benchmark.run check
conda run --no-capture-output -n dtb-vggt python scripts/preflight.py vggt
conda run --no-capture-output -n dtb-gsplat python scripts/preflight.py gsplat
conda run --no-capture-output -n dtb-anysplat python scripts/preflight.py anysplat

# 두 baseline의 pinned model weights를 미리 다운로드
conda run --no-capture-output -n dtb-vggt python scripts/prefetch.py --method all

bash scripts/download_data.sh
conda run --no-capture-output -n dtb-eval python -m benchmark.run prepare --scene all
```

[third_party.lock.json](third_party.lock.json)과 [.gitmodules](.gitmodules)의 pinned sources는 유지합니다. 현재 `check_pins()`가 AnySplat을 포함한 모든 upstream checkout을 검사하므로 모두 필요합니다. `bash scripts/setup.sh all`로 전체 설치를 수행할 수도 있습니다. Ours만 실행할 때는 AnySplat 환경 설치와 해당 preflight 검사를 생략할 수 있습니다. 모델과 matcher 가중치는 최초 사용 시에도 다운로드하며 기존 cache를 재사용합니다.

## Baseline 실행

두 baseline을 Kitchen/Room에서 순차 실행합니다. Ours는 아래 전용 entrypoint를 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh --run-id benchmark-v1
```

단일 baseline만 실행하려면 method를 지정합니다.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene kitchen --method vggt_3dgs --run-id kitchen-vggt-v1

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene room --method anysplat --run-id room-anysplat-v1

conda run --no-capture-output -n dtb-eval python scripts/make_table.py \
  --outputs outputs --csv outputs/results.csv
```

현재 AnySplat adapter는 32 training RGB만 camera prediction에 사용합니다. 공개 AnySplat evaluation의 target-RGB camera inference와는 다른 protocol입니다. 같은 split·primary test K·해상도·공통 metric으로 평가하되, 두 baseline의 native renderer와 연산 비용에는 차이가 있습니다.

## 최종 방법 재실행

먼저 VGGT source baseline을 생성하거나 완료된 기존 source를 재사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene kitchen --method vggt_3dgs --run-id final-source

PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n dtb-gsplat python -m benchmark.reproduce \
  --source-run outputs/vggt_3dgs/kitchen/final-source \
  --output-dir outputs/ours/kitchen/final-reproduction-v1
```

Room은 scene과 경로를 `room`으로 바꿉니다. Plan 확인은 `--dry-run`, camera refinement만 실행하려면 `--stage refine`을 사용합니다. 기존 matches, refined cameras 및 checkpoint 재사용 명령은 [재현 문서](docs/REPRODUCE_OURS.md)에 있습니다. 새 실행 디렉터리를 사용하며 기존 결과를 덮어쓰지 않습니다.

```bash
bash scripts/test.sh
```

## GitHub 배포 범위

[.gitignore](.gitignore)는 두 baseline과 최종 Ours에 필요한 first-party code/config를 지정합니다. `rgb_pose_refine.yaml`과 pose optimizer는 최종 staged 방법의 공통 설정·수학 함수로 사용되므로 포함합니다. 실행과 평가를 검증하는 테스트도 포함합니다.

MPR, Shared-K, Oracle, ablation/audit drivers와 연구 기록은 제외합니다. `data/`, `work/`, `outputs/`, `reports/`, 모델 가중치와 `main.tex`도 로컬에 보존하고 배포하지 않습니다. 기존 Git history에 포함된 파일은 ignore 규칙만으로 과거 commit에서 제거되지 않습니다.

이 저장소의 [MIT 라이선스](LICENSE)와 별개로, 외부 코드·모델·데이터셋의 라이선스를 확인하세요.
