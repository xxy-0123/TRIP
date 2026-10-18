# TRIP

PyTorch implementation of TRIP for time series forecasting with covariates. The repository includes the KITE benchmark framework, DAG, and other baselines.

## Installation

Use Python 3.10 and install the dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

Place the preprocessed datasets and `FORECAST_META.csv` in `dataset/forecasting/`. Data are not bundled with the code.

Series CSVs use the columns `date,data,cols`, with one contiguous block per variable and the target variable last. All variables must share the same chronological timestamps. Future covariate values are required over the forecast horizon.

For new datasets, initialize `FORECAST_META.csv` with the header `file_name`; the benchmark registers the series automatically.

## Training and evaluation

Run from the repository root:

```bash
bash scripts/TRIP.sh
```

```bash
bash scripts/DAG.sh
```

## Acknowledgments and license

The benchmark framework and baseline implementations are inherited from KITE. The upstream copyright notice, MIT license, and third-party attribution comments are retained.
