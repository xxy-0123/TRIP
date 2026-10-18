#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# The benchmark backend selects visible devices through --gpus.
gpu_args=()
if [ "${GPU:-0}" != cpu ]; then
    gpu_args=(--gpus "${GPU:-0}")
fi

# Optional arguments select datasets; by default run all 24 tasks.
datasets=("$@")
if [ "${#datasets[@]}" -eq 0 ]; then
    datasets=(NP PJM BE FR DE Energy Sdwpfh1 Sdwpfh2 Sdwpfm1 Sdwpfm2 Colbun Rapel)
fi

for dataset in "${datasets[@]}"; do
    case "$dataset" in
        Colbun|Rapel) horizons=(10 30) ;;
        NP|PJM|BE|FR|DE|Energy|Sdwpfh1|Sdwpfh2|Sdwpfm1|Sdwpfm2) horizons=(24 360) ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac
    for horizon in "${horizons[@]}"; do
        case "$horizon" in
            10) seq_len=60 ;;
            30) seq_len=180 ;;
            24) seq_len=168 ;;
            360) seq_len=720 ;;
        esac
        "${PYTHON:-python}" -B ./scripts/run_benchmark.py \
            --config-path rolling_forecast_config.json \
            --data-name-list "${dataset}.csv" \
            --strategy-args "{\"horizon\":${horizon},\"target_channel\":[-1]}" \
            --model-name trip.TRIP \
            --model-hyper-params "{\"seq_len\":${seq_len},\"horizon\":${horizon}}" \
            --seed 2021 \
            --eval-backend sequential \
            "${gpu_args[@]}" \
            --num-workers 1 \
            --timeout 60000 \
            --save-path "${dataset}/TRIP/H${horizon}"
    done
done
