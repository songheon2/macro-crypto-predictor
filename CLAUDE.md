# 암호화폐 불확실성 감소 모델 — 딥러닝 학습 파이프라인

> 목적: macro-collector가 생성한 전처리 데이터셋을 입력으로 받아
> 다수의 딥러닝 모델을 학습·평가·비교한다.

---

## 세션 시작 명령어

| 창 | 역할 | 시작 명령 |
|----|------|-----------|
| 창 1 | 설계 담당 | `claude "[설계 담당] CLAUDE.md를 읽고 설계 담당으로 시작해."` |
| 창 2 | 코딩 담당 | `claude "[코딩 담당] CLAUDE.md와 CLAUDE.local.md를 읽고 코딩 담당으로 시작해."` |
| 창 3 | 리뷰 담당 | `claude "[리뷰 담당] CLAUDE.md를 읽고 현재 구현된 파일을 리뷰해."` |

> **주의**: Claude Code가 파일 읽기 권한을 요청할 수 있습니다. 세션 시작 후 파일 접근 허용 여부를 확인하세요.

---

## 역할 정의

### [설계 담당] — 창 1
- **목적**: 아키텍처 설계 유지, 모듈 간 인터페이스 정의, 구현 규칙 관리
- **행동 규칙**:
  - 코드를 직접 작성하지 않는다. 설계와 지시만 한다.
  - 설계 변경 시 이 CLAUDE.md를 즉시 업데이트한다.
  - [리뷰 담당]의 제안은 반드시 [설계 담당]이 최종 판단한다.
    - 설계 의도에 부합 → CLAUDE.md 수정 후 [코딩 담당]에 변경 지시
    - 설계 의도와 불일치 → 기각, [코딩 담당]에 원래 설계 유지 지시
  - CLAUDE.md는 항상 **single source of truth**로 유지한다.
  - TBD 항목(타깃 변수, 평가 지표, 추가 모델)을 확정한 후 즉시 이 파일에 기재한다.
- **세션 시작 확인 문구**:
  > "설계 담당으로 시작합니다. CLAUDE.md 기준으로 현재 설계 상태를 요약할게요."

---

### [코딩 담당] — 창 2
- **목적**: CLAUDE.md 설계 기반으로 실제 코드 구현
- **행동 규칙**:
  - **가상환경 필수**: 세션 시작 시 `.venv`가 없으면 `python -m venv .venv`로 생성하고, 항상 가상환경을 활성화한 상태에서 패키지 설치·실행·테스트를 수행한다.
    - Windows: `.venv\Scripts\activate`
    - macOS/Linux: `source .venv/bin/activate`
  - 패키지 설치는 반드시 가상환경 내 `pip install -r requirements.txt`로만 한다.
  - 하이퍼파라미터·상수·경로 하드코딩 금지. 반드시 `config.py`에서만 가져온다.
  - 타입 힌트 필수, 로깅은 `logging` 모듈 사용 (`print` 금지).
  - 불명확한 사항은 CLAUDE.local.md에 질문으로 기록한다.
  - 파일 완성 시 CLAUDE.local.md 진행상황을 업데이트한다.
  - [설계 담당] 명시 지시 없이 `config.py` 임의 수정 금지.
  - [리뷰 담당]의 제안은 [설계 담당] 확인 없이 절대 반영하지 않는다.
  - **CLAUDE.md를 직접 수정하지 않는다.**
- **세션 시작 확인 문구**:
  > "코딩 담당으로 시작합니다. .venv 가상환경을 확인·생성하고 활성화한 후, CLAUDE.local.md에서 진행상황 확인 후 이어서 구현할게요."

---

### [리뷰 담당] — 창 3
- **목적**: 구현된 코드가 설계 의도에 맞는지 독립적으로 검토
- **행동 규칙**:
  - 코드를 직접 수정하지 않는다. 문제점과 개선 제안만 리포트한다.
  - CLAUDE.md 설계 기준과 실제 코드를 반드시 대조한다.
  - 리뷰 결과는 **`REVIEW.md`에 저장**한다. 형식: `[PASS]` / `[WARN]` / `[FAIL]` (파일명·위치·이유 명시)
  - 중점 검토: 데이터 리키지, 재현성(seed), 모델 저장·로드 정합성.
  - **CLAUDE.md를 직접 수정하지 않는다.**
- **세션 시작 확인 문구**:
  > "리뷰 담당으로 시작합니다. 현재 구현된 파일을 확인하고 REVIEW.md에 리뷰를 작성할게요."

---

## 역할 간 의사결정 흐름

```
[리뷰 담당] → REVIEW.md에 제안 기록 → [설계 담당] → 판단
                                            ├── 승인: CLAUDE.md 수정 → [코딩 담당]에 변경 지시
                                            └── 기각: [코딩 담당]에 원래 설계 유지 지시
```

> [코딩 담당]은 [설계 담당]의 지시 없이 리뷰 제안을 직접 반영하지 않는다.

---

## 프로젝트 구조

```
macro-crypto-predictor/
├── data/
│   ├── macro_features_train_scaled.csv   # macro-collector 산출물
│   ├── macro_features_val_scaled.csv
│   ├── macro_features_test_scaled.csv
│   └── scaler_params.json                # macro-collector 산출물 (μ, σ 저장)
├── src/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Dataset / DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py         # LSTM 모델
│   │   ├── transformer.py  # Transformer 모델
│   │   └── (추가 모델)
│   ├── trainer.py          # 학습 루프, 검증, 조기 종료
│   ├── evaluator.py        # 평가 지표 계산 및 비교
│   └── utils.py            # 시드 고정, 체크포인트 저장·로드
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_trainer.py
│   └── test_utils.py
├── outputs/
│   ├── checkpoints/        # 모델 가중치 저장
│   └── results/            # 평가 결과 CSV / 시각화
├── config.py               # 전체 상수·하이퍼파라미터 중앙 관리
├── train.py                # 학습 전용 진입점
├── evaluate.py             # 평가 전용 진입점
├── requirements.txt
├── CLAUDE.md               # 이 파일 (설계 명세, single source of truth)
├── CLAUDE.local.md         # 로컬 진행상황·질문 기록 (git 제외)
└── REVIEW.md               # 리뷰 결과 기록 (git 제외)
```

---

## 입력 데이터 명세

| 항목 | 내용 |
|------|------|
| 소스 | macro-collector 파이프라인 산출물 |
| 형식 | CSV (첫 번째 컬럼 = DatetimeIndex) |
| 분할 | train: 2021-01-01~2024-12-31 / val: 2025-01-01~2025-06-30 / test: 2025-07-01~2025-12-31 |
| 스케일 | StandardScaler 적용 완료 (train 기준 fit) |
| 피처 수 | 18개 (INPUT_SIZE = 18) |
| 타깃 변수 | `btc_open` (t+1) — 다음 행의 BTC 로그 차분 수익률 예측 (회귀) |
| 타깃 구성 | `df[TARGET_COL].shift(-1)` 후 마지막 행 NaN 제거, y = btc_open_{t+1} |

### 입력 피처 목록 (18개)

| # | 컬럼명 | 원본 의미 | 적용 변환 |
|---|--------|-----------|-----------|
| 1 | `open` | KOSPI 시가 | 로그 차분 |
| 2 | `volume` | KOSPI 거래량 | MA 대비 비율 |
| 3 | `gov_1y` | 국고채 1년 금리 | 단순 차분 |
| 4 | `usd_krw` | 달러/원 환율 | 로그 차분 |
| 5 | `gold_usd` | 금 가격 | 로그 차분 |
| 6 | `nasdaq_close` | 나스닥 종가 | 로그 차분 |
| 7 | `nasdaq_volume` | 나스닥 거래량 | MA 대비 비율 |
| 8 | `btc_open` | BTC 시가 수익률 | 로그 차분 |
| 9 | `btc_volume` | BTC 거래량 | MA 대비 비율 |
| 10 | `btc_rsi` | BTC RSI | 원본 유지 |
| 11 | `eth_open` | ETH 시가 수익률 | 로그 차분 |
| 12 | `eth_volume` | ETH 거래량 | MA 대비 비율 |
| 13 | `eth_rsi` | ETH RSI | 원본 유지 |
| 14 | `volatility_20d` | KOSPI 20일 변동성 | 원본 유지 |
| 15 | `term_spread_10y1y` | 장단기 스프레드 10Y-1Y | 단순 차분 |
| 16 | `term_spread_3y1y` | 장단기 스프레드 3Y-1Y | 단순 차분 |
| 17 | `credit_spread_aa` | 신용 스프레드 AA- | 단순 차분 |
| 18 | `credit_spread_bbb` | 신용 스프레드 BBB- | 단순 차분 |

> ⚠️ **리키지 주의**: `btc_open`은 피처(#8)이자 타깃 컬럼이다.
> 피처의 `btc_open[t]` = `log(P_t/P_{t-1})`, 타깃의 `btc_open[t]` = `log(P_{t+1}/P_t)` 로 **값이 다르다**.
> [코딩 담당]은 반드시 X 배열 구성 **완료 후** shift(-1)로 y를 별도 생성해야 한다.

---

## 모델 명세

| 모델 | 파일 | 비고 |
|------|------|------|
| LSTM | `src/models/lstm.py` | 기본 베이스라인 |
| Transformer | `src/models/transformer.py` | 시계열 어텐션 |
| (추가 모델) | — | [설계 담당] 지시 후 추가 |

---

## `config.py` 포함 항목

```python
# 경로
DATA_DIR: str = "data"
OUTPUT_DIR: str = "outputs"
CHECKPOINT_DIR: str = "outputs/checkpoints"
RESULTS_DIR: str = "outputs/results"
SCALER_PARAMS_PATH: str = "data/scaler_params.json"  # macro-collector 제공 μ, σ

# 학습
SEED: int = 42
BATCH_SIZE: int = 64
EPOCHS: int = 100
LEARNING_RATE: float = 1e-3
EARLY_STOPPING_PATIENCE: int = 10
SEQ_LEN: int = 30          # 입력 시퀀스 길이 (타임스텝)
INPUT_SIZE: int = 18       # 입력 피처 수 (고정, 변경 시 [설계 담당] 승인 필요)

# 모델 — LSTM
LSTM_HIDDEN_SIZE: int = 128
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2

# 모델 — Transformer
TRANSFORMER_D_MODEL: int = 64
TRANSFORMER_NHEAD: int = 4
TRANSFORMER_NUM_LAYERS: int = 2
TRANSFORMER_DROPOUT: float = 0.1

# 평가
TARGET_COL: str = "btc_open"   # 타깃 컬럼명 — 다음 행 시가 예측 (shift(-1) 적용)
```

---

## 모듈별 역할 및 인터페이스

### `config.py`
- **역할**: 모든 상수·하이퍼파라미터 중앙 관리
- **규칙**: 단순 상수로 작성, 다른 모듈이 `import config`로 참조

### `src/dataset.py`
- **역할**: CSV 로드 → 슬라이딩 윈도우 시퀀스 생성 → DataLoader 반환
- **CSV 로드 규칙**: `pd.read_csv(path, index_col=0, parse_dates=True)` — index_col=0과 parse_dates=True 필수
- **전처리 순서** (순서 엄수):
  1. CSV 로드
  2. 초반 연속 NaN 블록 제거 — 모든 컬럼이 처음으로 유효한 행을 찾아 그 이후부터 슬라이싱
     ```python
     first_valid = df.dropna(how="any").index[0]
     df = df.loc[first_valid:]
     ```
     (중간에 산발적으로 남은 NaN은 별도 처리하지 않음 — 전처리 완료 데이터 기준)
  3. `df[TARGET_COL].shift(-1)` 으로 y 컬럼 생성
  4. shift로 생긴 마지막 행 NaN 제거 (`iloc[:-1]`)
  5. 슬라이딩 윈도우 구성
- **슬라이딩 윈도우**: `X = features[i : i+SEQ_LEN]`, `y = target[i+SEQ_LEN-1]`
  - 즉, y는 시퀀스 마지막 타임스텝의 다음 행 btc_open 값 (= btc_open_{t+1})
- **핵심 클래스/함수**:
  - `MacroDataset(Dataset)` — `__getitem__`이 `(seq, label)` 튜플 반환
  - `get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]` — train/val/test 순서로 반환

### `src/models/lstm.py`
- **역할**: LSTM 기반 시계열 모델
- **핵심 클래스**:
  - `LSTMModel(nn.Module)` — `forward(x: Tensor) -> Tensor`

### `src/models/transformer.py`
- **역할**: Transformer 기반 시계열 어텐션 모델
- **핵심 클래스**:
  - `TransformerModel(nn.Module)` — `forward(x: Tensor) -> Tensor`

### `src/trainer.py`
- **역할**: 학습 루프, val loss 기반 조기 종료, 체크포인트 저장
- **핵심 클래스**:
  - `Trainer` — `train(model, train_loader, val_loader) -> None`
  - 에폭별 train/val loss 로깅
  - val loss 최소 시점에 체크포인트 자동 저장

### `src/evaluator.py`
- **역할**: 저장된 체크포인트 로드 → test 셋 평가 → 모델 간 비교 결과를 이미지로 저장
- **핵심 클래스**:
  - `Evaluator` — `evaluate(model_name: str, checkpoint_path: str) -> dict`
- **역변환 적용**:
  - RMSE는 z-score 단위로 계산 (모델 간 비교 기준)
  - 차트 출력은 `invert_target()` 적용 후 **% 수익률 단위**로 표시
  - `load_scaler_params(config.SCALER_PARAMS_PATH, config.TARGET_COL)` 로 btc_open μ, σ 추출
- **출력 형식**: PNG 이미지 (`outputs/results/` 저장)
  - `prediction_vs_actual_{model_name}.png` — 예측값 vs 실제값 시계열 차트 (y축: % 수익률)
  - `model_comparison_rmse.png` — 모델별 RMSE 비교 바 차트 (z-score 단위)
- **평가 지표**: RMSE (Root Mean Squared Error) — z-score 단위, 회귀 기준 지표

### `src/utils.py`
- **역할**: 재현성 보장, 체크포인트 I/O, 역변환
- **핵심 함수**:
  - `set_seed(seed: int) -> None` — torch·numpy·random 전체 시드 고정
  - `save_checkpoint(model: nn.Module, path: str, epoch: int, val_loss: float) -> None`
  - `load_checkpoint(model: nn.Module, path: str) -> dict`
  - `load_scaler_params(path: str, col: str) -> dict` — scaler_params.json에서 col 컬럼의 μ, σ 추출 후 `{"mean": float, "std": float}` 반환
    - 실제 파일 형식: `{"columns": [...], "mean": [...], "scale": [...]}` (18개 컬럼 전체 배열)
    - col 파라미터로 columns 배열에서 인덱스 탐지 → mean[idx], scale[idx] 추출
  - `invert_target(z: np.ndarray, mean: float, std: float) -> np.ndarray` — z-score → % 수익률 역변환
    ```
    1단계: log_return = z * std + mean   (inverse StandardScaler)
    2단계: pct_return = exp(log_return) - 1  (inverse log diff)
    ```

### `train.py`
- **역할**: 학습 전용 진입점
- **실행 순서**:
  1. config 로드
  2. `set_seed(config.SEED)`
  3. `get_dataloaders()` 호출
  4. 모델 초기화
  5. `Trainer.train()` 실행

### `evaluate.py`
- **역할**: 평가 전용 진입점
- **실행 순서**:
  1. config 로드
  2. 체크포인트 로드
  3. `Evaluator.evaluate()` 실행
  4. 결과 저장

---

## 실행 방식

```bash
# 최초 1회: 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 학습
python train.py

# 평가
python evaluate.py
```

---

## 구현 순서 ([코딩 담당] 참고)

[코딩 담당]은 아래 순서를 준수해 구현한다:

1. 가상환경 생성 및 활성화 (`python -m venv .venv`)
2. `requirements.txt`
3. 의존성 설치 (`pip install -r requirements.txt`)
4. `config.py`
5. `src/__init__.py`, `src/models/__init__.py`
6. `src/utils.py` + `tests/test_utils.py`
7. `src/dataset.py` + `tests/test_dataset.py`
8. `src/models/lstm.py`
9. `src/models/transformer.py`
10. `src/trainer.py` + `tests/test_trainer.py`
11. `src/evaluator.py`
12. `train.py`
13. `evaluate.py`

> 순서를 변경할 경우 [설계 담당]이 이 목록을 업데이트하고 이유를 기록한다.

---

## 코딩 규칙

| 규칙 | 내용 |
|------|------|
| 하이퍼파라미터·상수·경로 | `config.py`에서만 관리, 모듈 내 하드코딩 금지 |
| `config.py` 수정 | [설계 담당] 명시 지시 후에만 허용 |
| 재현성 | 학습 시작 시 반드시 `set_seed(config.SEED)` 호출 |
| 로깅 | `print` 대신 Python `logging` 모듈 사용 |
| 타입 힌트 | 모든 함수 시그니처에 필수 |
| 모델 저장 | 에폭별 val loss 최소 시점 체크포인트 저장 |
| 데이터 리키지 | test 셋은 최종 평가 시에만 사용. 하이퍼파라미터 튜닝에 사용 금지 |
| 모델 인터페이스 | 모든 모델은 `forward(x: Tensor) -> Tensor` 인터페이스 통일 |
| 에러 처리 | 예외는 명시적으로 처리, 조용한 실패(silent fail) 금지 |
| 테스트 | 핵심 함수는 `tests/`에 단위 테스트 작성, 구현과 함께 커밋 |

---

## 채워야 할 TBD 항목 ([설계 담당] 확정 필요)

1. ~~**타깃 변수**~~ ✅ **확정**: `btc_open` (t+1) — shift(-1) 적용, 회귀 문제
2. ~~**평가 지표**~~ ✅ **확정**: RMSE — 회귀 기준 지표
3. **추가 모델** — LSTM/Transformer 외 고려 모델

> TBD 항목 확정 후 이 파일의 해당 섹션(`evaluator.py` 평가 지표)을 즉시 업데이트한다.

---

## 설계 변경 이력

| 날짜 | 변경 내용 | 이유 | 결정자 |
|------|-----------|------|--------|
| 2026-06-29 | 초기 설계 작성 | 프로젝트 시작 | [설계 담당] |
| 2026-06-29 | 타깃 변수 확정: `btc_open` (shift(-1), 회귀) | 사용자 결정 | [설계 담당] |
| 2026-06-29 | 평가 지표 확정: RMSE | 사용자 결정 | [설계 담당] |
| 2026-06-29 | 피처 목록 확정 (18개), INPUT_SIZE=18 추가, btc_open 리키지 주의사항 명시 | 피처 목록 공유 | [설계 담당] |
| 2026-06-29 | evaluator 출력 형식 CSV → PNG 이미지로 변경 | 결과 시각화 직관성 확보 | [설계 담당] |
| 2026-06-29 | 역변환 기능 설계 추가: z-score → 로그수익률 → % 수익률 | 모델 예측값을 실제 해석 가능한 수익률로 변환 | [설계 담당] |
| 2026-06-29 | scaler_params.json 형식 명세 수정: 단일값 → 18개 컬럼 배열 구조, load_scaler_params에 col 파라미터 추가 | 실제 파일 형식이 설계 명세와 달랐음 (코딩 담당 보고) | [설계 담당] |
| 2026-06-29 | 데이터 형식 Parquet → CSV 변경 | 규모 대비 CSV가 충분하고 디버깅 편의성 높음 | [설계 담당] |
| 2026-06-29 | dataset.py 전처리 순서 확정: 초반 연속 NaN 블록만 제거 후 shift→윈도우 | 고정 일수 아닌 동적 유효 시작일 탐지 방식 채택 | [설계 담당] |

---

## CLAUDE.local.md 운영 규칙

- [코딩 담당]이 단독으로 관리하는 로컬 파일
- git에 커밋하지 않는다 (`.gitignore`에 추가)
- 포함 내용:
  - 현재 구현 진행상황 (완료 / 진행 중 / 미착수)
  - [설계 담당]에게 묻고 싶은 질문
  - 임시 메모 및 트러블슈팅 기록

---

## REVIEW.md 운영 규칙

- [리뷰 담당]이 단독으로 관리하는 리뷰 결과 파일
- git에 커밋하지 않는다 (`.gitignore`에 추가)
- 포함 내용:
  - 리뷰 일시 및 대상 파일 목록
  - `[PASS]` / `[WARN]` / `[FAIL]` 항목 (파일명·위치·이유 명시)
  - [설계 담당]에 전달할 개선 제안 요약

---

## .gitignore 필수 항목

```
CLAUDE.local.md
REVIEW.md
.venv/
outputs/
data/
__pycache__/
*.pyc
```
