# CaimanGUI

[CaImAn](https://github.com/flatironinstitute/CaImAn)을 위한 manual curation tool

## 설치

```bash
conda create -n caiman caiman
conda activate caiman
pip install git+https://github.com/nctrl-lab/CaimanGUI
```

## 실행

```bash
conda activate caiman
caiman
```

## 화면 구성

<p align="center" width=100%>
  <img src="images/GUI_main.png" width="90%">
</p>

- **왼쪽 상단** — FOV (neuron 위치)
- **왼쪽 하단** — Fluorescence trace
- **오른쪽** — Neuron 테이블 및 파라미터

## 단축키

### 파일

| 키 | 동작 |
|----|------|
| `Ctrl+O` | HDF5 열기 |
| `Ctrl+S` | 저장 |
| `Ctrl+Shift+S` | 다른 이름으로 저장 |
| `Ctrl+R` | CaImAn Runner 실행 |
| `Ctrl+W` | 종료 |

### Neuron 분류

| 키 | 동작 |
|----|------|
| `Alt+G` | Good |
| `Alt+N` | Noise |
| `Alt+M` | Uncertain |
| `Up/Down` | 다른 unit 선택 |
| `ESC` | 선택 해제 |

* Dragging하면 여러 unit 선택 가능

## Neuron 테이블

| 열 | 설명 |
|----|------|
| **ID** | Neuron 번호 |
| **Rval** | Spatial correlation (spatial quality) |
| **SNR** | Signal-to-noise ratio (temporal quality) |
| **CNN** | CNN 예측 점수 (spatial quality) |
| **Area** | Spatial footprint 크기 (pixel) |
| **Quality** | CaImAn 자동 분류 |
| **Status** | 사용자 지정 상태 |

- 열 헤더 클릭으로 정렬
- "View components"로 상태별 필터링
- `Ctrl`/`Shift` 클릭으로 다중 선택
- 왼쪽 아래 Quality parameter를 수정하면 Quality unit 들이 바뀐다

## 보기 모드

| 모드 | 설명 |
|------|------|
| **reset** | Contour 표시 |
| **neurons** | 선택된 neuron들을 colormap으로 표시 |
| **correlation** | 선택한 neuron과 temporal correlation이 높은 neuron 표시 |
| **accepted** | Good neuron만 표시, 방향키 탐색 |
| **neighbors** | 인접 neuron 간 correlation 표시 |

## Quality parameters

모든 low 임계값 통과(AND) + high 임계값 하나 이상 통과(OR) → Quality "Good".

## Pipeline 실행 (`Ctrl+R`)

<p align="center" width=100%>
  <img src="images/GUI_runner.png" width="90%">
</p>

1. 데이터 폴더 선택 (AVI/TIFF)
2. Pipeline 유형 선택 — 1p (CNMF-E) / 2p (CNMF)
3. 파라미터 조정
4. Preview → Run

## Outcome variables (model.estimates)

| 변수 | 설명 |
|------|------|
| `accepted_list` | Good neuron ID |
| `rejected_list` | Noise neuron ID |
| `uncertain_list` | Uncertain neuron ID |
| `idx_components` | 품질 필터 통과 neuron |
| `idx_components_bad` | 품질 필터 미통과 neuron |

HDF5 형식으로 저장.