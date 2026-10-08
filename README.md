# Digital Twin Benchmark

Mip-NeRF 360의 `kitchen`, `room` 장면에서 **VGGT + 3DGS(gsplat)**와 **AnySplat**을 동일한 조건으로 비교하는 벤치마크입니다. 각 방법의 재구성·렌더링은 원래 구현을 사용하고, 생성한 RGB PNG를 공통 평가기로 평가합니다.

## 평가 조건

장면마다 학습 이미지 32장과 테스트 이미지 4장을 사용합니다.

| 장면 | 이미지 범위(양 끝 포함) | 테스트 이미지 |
|---|---|---|
| kitchen | DSCF0656–DSCF0691 | DSCF0664, DSCF0673, DSCF0682, DSCF0691 |
| room | DSCF4667–DSCF4702 | DSCF4675, DSCF4684, DSCF4693, DSCF4702 |

- 두 방법 모두 동일한 학습 이미지와 448×448 리사이즈·중앙 크롭 전처리를 사용합니다.
- 학습 카메라 중심으로 Sim(3) 정렬을 계산하고, 같은 변환을 참조 테스트 카메라에 적용합니다.
- 테스트 RGB는 재구성과 정렬에 사용하지 않습니다. 테스트 카메라의 참조 보정 정보는 사용합니다.
- 테스트 이미지 4장의 PSNR, SSIM, LPIPS와 실행 시간을 기록합니다.
- VGGT + 3DGS는 기본 30,000회 최적화하며, AnySplat은 추가 최적화 없이 추론합니다.

## 1. GPU 환경 설치

Linux x86_64, conda, NVIDIA GPU, 호환 드라이버, **CUDA 12.1 툴킷(`nvcc` 포함)**과 C++ 컴파일러가 필요합니다.

저장소 루트에서 아래 순서대로 실행하세요. 설치 스크립트는 서브모듈 초기화와 실행 환경·의존성 설치를 자동으로 처리합니다. 각 단계가 성공한 뒤 다음 단계로 진행하세요.

```bash
# 사용자 공용 Python 패키지와 실행 환경 분리
export PYTHONNOUSERSITE=1

# 환경 생성 및 의존성 설치
bash scripts/setup.sh all

# 환경 확인
conda run --no-capture-output -n dtb-eval python -m benchmark.run check
conda run --no-capture-output -n dtb-vggt python scripts/preflight.py vggt
conda run --no-capture-output -n dtb-gsplat python scripts/preflight.py gsplat
conda run --no-capture-output -n dtb-anysplat python scripts/preflight.py anysplat

# 모델 가중치 다운로드
conda run --no-capture-output -n dtb-vggt python scripts/prefetch.py
```

`PYTHONNOUSERSITE` 설정은 현재 터미널에 적용됩니다. 새 터미널에서 아래 명령들을 실행할 때도 먼저 `export PYTHONNOUSERSITE=1`을 실행하세요. 설치 스크립트와 전체 실행 스크립트는 내부에서도 이 설정을 적용합니다.

설치 중 `nvcc is not on PATH`가 나오면 설치된 CUDA 툴킷의 `bin` 경로를 추가한 뒤 설치를 다시 실행하세요. 아래 경로는 실제 설치 위치에 맞게 변경합니다.

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash scripts/setup.sh all
```

`EnvironmentLocationNotFound`는 실행 환경이 아직 생성되지 않았다는 뜻입니다. 설치 오류를 해결하고 `setup.sh all`을 완료한 뒤 환경 확인 명령을 실행하세요.

소스 버전은 [third_party.lock.json](third_party.lock.json), 모델 버전은 [configs/methods/](configs/methods/)에 고정되어 있습니다. 모델 캐시용 디스크 공간과 학습 이미지 32장을 처리할 GPU 메모리를 확보하세요.

## 2. 데이터 준비

아래 스크립트로 [공식 Mip-NeRF 360 데이터셋](https://jonbarron.info/mipnerf360/)을 다운로드하고 `kitchen`, `room`의 이미지와 카메라 파일을 자동 배치합니다. `curl`과 Python 3이 필요합니다.

```bash
bash scripts/download_data.sh
```

공식 ZIP은 약 12.5GB이며 `data/downloads/`에 보관됩니다. 압축 해제 공간도 필요합니다. 다운로드가 중단되면 같은 명령으로 이어받을 수 있고, 이미 배치된 파일은 유지합니다.

배치 결과는 다음과 같습니다.

```text
data/mipnerf360/
  kitchen/
    images/DSCF0656.JPG ...
    sparse/0/cameras.bin
    sparse/0/images.bin
  room/
    images/DSCF4667.JPG ...
    sparse/0/cameras.bin
    sparse/0/images.bin
```

COLMAP 바이너리 형식의 `PINHOLE` 또는 `SIMPLE_PINHOLE` 카메라만 지원하며, 참조 `points3D.bin`은 사용하지 않습니다. 경로가 다르면 [configs/scenes/](configs/scenes/)를 수정하거나 `--data-root`를 지정하세요.

```bash
conda run --no-capture-output -n dtb-eval python -m benchmark.run prepare --scene all
```

전처리 결과는 `work/prepared/`에 저장됩니다. 이미 전처리를 완료했다면 다음 실행 단계로 진행하세요. 데이터나 전처리 설정을 변경할 때는 새 `prepared_root`를 지정하세요.

## 3. 벤치마크 실행

전체 비교는 아래 명령 하나로 실행합니다.

GPU가 할당된 서버·세션에서 실행하세요. 설치 단계의 세 가지 `preflight.py` 검사가 모두 통과해야 합니다. `CUDA is unavailable`이 나오면 현재 세션의 GPU 접근과 해당 환경의 CUDA 지원을 확인하세요.

```bash
# 두 방법 × 두 장면 전체 실행
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh --run-id benchmark-v1
```

특정 방법·장면만 실행하려면 아래 예시 중 하나를 사용하세요.

```bash
# VGGT + 3DGS 단일 실행
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene kitchen --method vggt_3dgs --run-id kitchen-vggt-v1

# AnySplat 단일 실행
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dtb-eval python -m benchmark.run run \
  --scene room --method anysplat --run-id room-anysplat-v1
```

전체 실행은 GPU 한 장에서 네 조합을 순차적으로 처리합니다. `CUDA_VISIBLE_DEVICES`로 사용할 GPU를 선택하세요. `run`은 전처리가 없으면 자동으로 수행하며, 기존 실행 디렉터리는 덮어쓰지 않습니다. 실패한 실행의 로그를 확인한 뒤 새 `--run-id`로 다시 실행하세요.

## 4. 결과 확인

```bash
conda run --no-capture-output -n dtb-eval python scripts/make_table.py \
  --outputs outputs --csv outputs/results.csv
```

결과는 `outputs/<method>/<scene>/<run-id>/`에 저장됩니다.

| 파일·폴더 | 내용 |
|---|---|
| `renders/` | 테스트 이미지 4장의 렌더링 결과 |
| `metrics.json` | 이미지별·평균 PSNR, SSIM, LPIPS |
| `timing.json` | 실행 시간 |
| `alignment.json` | 카메라 정렬 결과 |
| `resolved_config.json` | 실행에 사용한 설정 |
| `status.json`, `logs/` | 실행 상태와 로그 |
| `native/` | 각 방법의 체크포인트 등 |

공통 설정은 [configs/benchmark.yaml](configs/benchmark.yaml)에서 변경할 수 있습니다. 결과를 비교할 때는 정렬 방식, 해상도, 하드웨어와 시간 측정 조건을 맞추세요.

다운로드 ZIP과 원본 데이터(`data/`), 전처리·설치 중간 파일(`work/`), 실험 결과(`outputs/`)는 Git 추적에서 제외됩니다.

GPU 전체 실행의 검증 이력은 [VALIDATION.md](VALIDATION.md)를 참고하세요.

이 저장소의 MIT 라이선스와 별개로, 외부 코드·모델·데이터셋의 라이선스를 확인하세요.
