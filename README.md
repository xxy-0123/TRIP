# TRIP

PyTorch implementation of TRIP for time series forecasting with supplied future covariates. The repository includes the KITE benchmark framework, DAG, and the other baseline implementations distributed with that framework.

## Installation

Use Python 3.10 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The TRIP training protocol uses PyTorch 2.4.1 and SciPy 1.10.1. CPU and CUDA execution are supported.

## Data

Place the preprocessed benchmark data in `dataset/forecasting/`, including `FORECAST_META.csv`. Data are not bundled with the code.

Each series CSV uses the benchmark's long format with columns `date,data,cols`. Store all timestamps for each variable in a contiguous block, use the same chronological timestamps for every variable, and put the target variable last. The experiment scripts select the last variable as the target and use the remaining variables as covariates. Future covariate values must be supplied over the complete forecast horizon.

For a new collection of files, create `FORECAST_META.csv` with a single header line, `file_name`. The benchmark automatically registers the series on its first run.

| Datasets | Covariates | Short task `(L, H)` | Long task `(L, H)` |
| --- | ---: | --- | --- |
| NP, PJM, BE, FR, DE | 2 | (168, 24) | (720, 360) |
| Energy | 5 | (168, 24) | (720, 360) |
| Sdwpfh1, Sdwpfh2, Sdwpfm1, Sdwpfm2 | 6 | (168, 24) | (720, 360) |
| Colbun, Rapel | 2 | (60, 10) | (180, 30) |

## Training and evaluation

Run one task from the repository root:

```bash
python -B scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list Colbun.csv \
  --strategy-args '{"horizon":10,"target_channel":[-1]}' \
  --model-name trip.TRIP \
  --model-hyper-params '{"seq_len":60,"horizon":10}' \
  --seed 2021 --eval-backend sequential --gpus 0 --num-workers 1 \
  --save-path Colbun/TRIP/H10
```

Run all 24 tasks, or select datasets:

```bash
bash scripts/w_future/TRIP.sh
bash scripts/w_future/TRIP.sh Colbun Rapel
```

The launcher runs tasks sequentially on GPU 0 when CUDA is available, falling back to CPU otherwise. Use `GPU=1 bash scripts/w_future/TRIP.sh` to select another GPU, or `GPU=cpu bash scripts/w_future/TRIP.sh` to force CPU execution. Set `PYTHON` to select a Python executable. For direct benchmark commands, omit `--gpus` to run on CPU. Reports are written under `result/`.

The default split is 70% training, 10% validation and 20% testing, in chronological order. Scalers are fitted on the training split. Evaluation uses rolling origins with stride 1 and at most 48,000 windows. Training uses seed 2021.

Stage 1 uses Adam with learning rate 0.001, batch size 128, at most 100 epochs and patience 10, selecting the earliest checkpoint with the lowest validation MAE. Stage 2 freezes the backbone and fits `2C + 11` adaptation parameters with L-BFGS-B, using at most 30 iterations and 60 objective/gradient evaluations. The neutral initialization and accepted iterates are compared by validation MAE.

Baseline launchers are available in `scripts/w_future/`, including `KITE.sh` and `DAG.sh`.

## Implementation and parameter count

- `ts_benchmark/baselines/trip/models/trip_model.py`: target projections, shared covariate response, correction decoder and adaptation parameters.
- `ts_benchmark/baselines/trip/trip.py`: benchmark adapter and Stage 1 training.
- `ts_benchmark/baselines/trip/routing.py`: Stage 2 fitting.

Count all model parameters, including the frozen backbone and Stage 2 parameters:

```bash
python -B scripts/count_trip_parameters.py
```

With `K = 24` and `R = min(8, L, H)`, the total is

```text
P = H(L + K + 3) + R(L + H) + 18C + 189.
```

Short12 spans 1,655–6,513 parameters; Long12 spans 8,115–277,857. The Full24 geometric mean is 28,200.776 parameters.

## Acknowledgments and license

The benchmark infrastructure and baseline implementations are inherited from KITE. Their original copyright notice and MIT license are retained in `LICENSE`; attribution comments in third-party source files are also retained.
