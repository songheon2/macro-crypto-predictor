# 암호화폐 불확실성 감소 모델 — 딥러닝 학습 파이프라인

거시경제 지표를 기반으로 BTC 다음 날 시가 수익률을 예측하는 딥러닝 파이프라인입니다.
[macro-collector](https://github.com/songheon2/macro-collector) 프로젝트의 전처리 산출물을 입력으로 받아 LSTM·Transformer 모델을 학습·평가·비교합니다.

---

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [입력 데이터](#입력-데이터)
- [모델](#모델)
- [설치 및 실행](#설치-및-실행)
- [결과](#결과)
- [설계 원칙](#설계-원칙)

---

## 프로젝트 구조

```
macro-crypto-predictor/
├── data/
│   ├── macro_features_train_scaled.csv   # macro-collector 산출물
│   ├── macro_features_val_scaled.csv
│   ├── macro_features_test_scaled.csv
│   └── scaler_params.json                # btc_open StandardScaler μ, σ
├── src/
│   ├── dataset.py       # PyTorch Dataset / DataLoader (슬라이딩 윈도우)
│   ├── models/
│   │   ├── lstm.py      # LSTM 모델
│   │   └── transformer.py  # Transformer 모델
│   ├── trainer.py       # 학습 루프, 조기 종료, 체크포인트 저장
│   ├── evaluator.py     # RMSE 평가, 결과 시각화
│   └── utils.py         # 시드 고정, 체크포인트 I/O, 역변환
├── tests/               # 단위 테스트
├── outputs/
│   ├── checkpoints/     # 모델 가중치 (.pt)
│   └── results/         # 평가 결과 이미지 (.png)
├── config.py            # 전체 상수·하이퍼파라미터 중앙 관리
├── train.py             # 학습 진입점
└── evaluate.py          # 평가 진입점
```

---

## 입력 데이터

| 항목 | 내용 |
|------|------|
| 형식 | CSV (DatetimeIndex) |
| 분할 | train: 2021-01-01~2024-12-31 / val: 2025-01-01~2025-06-30 / test: 2025-07-01~2025-12-31 |
| 스케일 | StandardScaler 적용 완료 (train 기준 fit) |
| 피처 수 | 18개 |
| 타깃 | `btc_open` (t+1) — 다음 날 BTC 시가 로그 수익률 (회귀) |

### 입력 피처 (18개)

| # | 컬럼 | 의미 | 변환 |
|---|------|------|------|
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

### 타깃 역변환

모델은 z-score를 예측합니다. 실제 수익률로 해석하려면 아래 순서로 역변환합니다.

```
모델 예측값 (z-score)
  → z * σ + μ            (inverse StandardScaler) → 로그 수익률
  → exp(log_return) - 1  (inverse log diff)        → % 수익률
```

`data/scaler_params.json` 형식 (18개 컬럼 전체 배열):
```json
{
  "columns": ["open", "volume", ..., "btc_open", ..., "credit_spread_bbb"],
  "mean":    [0.0003, 1.0021, ..., 0.0012,  ..., 0.0005],
  "scale":   [0.0251, 0.1832, ..., 0.0381,  ..., 0.0012]
}
```

---

## 모델

### LSTM
- 순차적 시계열 패턴 학습
- hidden_size=128, layers=2, dropout=0.2

### Transformer
- Self-Attention으로 장거리 의존성 포착
- d_model=64, heads=4, layers=2, dropout=0.1

공통: 입력 시퀀스 30일 → 다음 날 btc_open 예측 (1개 값 출력)

### 현재 평가 결과 (test set)

| 모델 | RMSE (z-score) |
|------|----------------|
| LSTM | 0.5019 |
| Transformer | 0.5241 |

> 나이브 베이스라인(항상 0 예측) RMSE ≈ 1.0 기준, 두 모델 모두 약 50% 개선

---

## 설치 및 실행

### 1. 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 데이터 준비

`data/` 디렉토리에 아래 파일을 배치합니다. (macro-collector 프로젝트 산출물)

```
data/macro_features_train_scaled.csv
data/macro_features_val_scaled.csv
data/macro_features_test_scaled.csv
data/scaler_params.json
```

### 3. 학습

```bash
python train.py
```

- 학습 완료 시 `outputs/checkpoints/{model_name}_best.pt` 저장
- val loss 기준 조기 종료 적용 (patience=10)

### 4. 평가

```bash
python evaluate.py
```

- `outputs/results/prediction_vs_actual_{model_name}.png` — 예측 vs 실제 시계열 차트
- `outputs/results/model_comparison_rmse.png` — 모델별 RMSE 비교 차트

### 5. 테스트

```bash
pytest tests/
```

---

## 주요 하이퍼파라미터

`config.py`에서 관리합니다.

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `SEED` | 42 | 재현성 시드 |
| `BATCH_SIZE` | 64 | 배치 크기 |
| `EPOCHS` | 100 | 최대 학습 에폭 |
| `LEARNING_RATE` | 1e-3 | Adam 학습률 |
| `EARLY_STOPPING_PATIENCE` | 10 | 조기 종료 patience |
| `SEQ_LEN` | 30 | 입력 시퀀스 길이 (일) |
| `INPUT_SIZE` | 18 | 입력 피처 수 |

---

## 설계 원칙

- 모든 하이퍼파라미터·경로는 `config.py`에서만 관리
- test 셋은 최종 평가 시에만 사용 (데이터 리키지 방지)
- 학습 재현성 보장: `set_seed(config.SEED)` 필수 호출
- 모든 모델은 `forward(x: Tensor) -> Tensor` 인터페이스 통일
