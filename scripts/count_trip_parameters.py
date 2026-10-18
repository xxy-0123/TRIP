"""Count all TRIP parameters, including frozen Stage 1 weights."""

from pathlib import Path
from types import SimpleNamespace
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_benchmark.baselines.trip.models.trip_model import TRIPModel


def main():
    channels = {
        "NP": 2,
        "PJM": 2,
        "BE": 2,
        "FR": 2,
        "DE": 2,
        "Energy": 5,
        "Sdwpfh1": 6,
        "Sdwpfh2": 6,
        "Sdwpfm1": 6,
        "Sdwpfm2": 6,
        "Colbun": 2,
        "Rapel": 2,
    }
    totals = {"Short12": [], "Long12": []}
    print("dataset,setting,L,H,C,total_parameters,stage2_parameters")
    for dataset, c in channels.items():
        sizes = (
            ((60, 10), (180, 30))
            if dataset in ("Colbun", "Rapel")
            else ((168, 24), (720, 360))
        )
        for setting, (length, horizon) in zip(totals, sizes):
            model = TRIPModel(
                SimpleNamespace(seq_len=length, pred_len=horizon, input_dim=c + 1)
            )
            total = sum(p.numel() for p in model.parameters())
            stage2 = sum(p.numel() for _, p in model.adaptation_parameters())
            rank = min(8, length, horizon)
            expected = (
                horizon * (length + 24 + 3) + rank * (length + horizon) + 18 * c + 189
            )
            if total != expected or stage2 != 2 * c + 11:
                raise RuntimeError(
                    f"Unexpected parameter count for {dataset}, H={horizon}"
                )
            totals[setting].append(total)
            print(f"{dataset},{setting},{length},{horizon},{c},{total},{stage2}")
    for setting, values in totals.items():
        print(f"{setting} range: {min(values):,}–{max(values):,}")
    values = totals["Short12"] + totals["Long12"]
    geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
    print(f"Full24 geometric mean: {geometric_mean:,.3f}")


if __name__ == "__main__":
    main()
