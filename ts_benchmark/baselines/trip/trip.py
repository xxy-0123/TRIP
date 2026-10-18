"""Benchmark adapter for two-stage TRIP training and rolling forecasts."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..deep_forecasting_model_base import DeepForecastingModelBase
from .models.trip_model import TRIPModel
from .routing import fit_adaptation
from ts_benchmark.utils.random_utils import fix_random_seed

MODEL_HYPER_PARAMS = {
    "norm": True,
    "lr": 0.001,
    "batch_size": 128,
    "num_epochs": 100,
    "patience": 10,
    "num_workers": 0,
    "loss": "window_mae",
    "lradj": "constant",
    "inference_batch_size": 1024,
    "use_amp": 0,
    "fusion_method": "",
    "device": None,
}


class _Windows(Dataset):
    def __init__(self, y, x, length, horizon, first, last):
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.history = torch.cat((self.y, self.x), dim=1)
        self.length, self.horizon = (length, horizon)
        self.first, self.last = (first, last)

    def __len__(self):
        return max(0, self.last - self.first + 1)

    def __getitem__(self, index):
        origin = self.first + index
        return (
            self.history[origin - self.length : origin],
            self.y[origin : origin + self.horizon],
            self.x[origin : origin + self.horizon],
        )


class TRIP(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        if "pred_len" in kwargs:
            if "horizon" in kwargs and kwargs["horizon"] != kwargs["pred_len"]:
                raise ValueError("horizon and pred_len must match")
            kwargs["horizon"] = kwargs["pred_len"]
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)
        c = self.config
        if not c.norm or c.loss != "window_mae" or c.lradj != "constant":
            raise ValueError("TRIP requires norm=True, window_mae and constant LR")
        if c.num_workers != 0 or c.use_amp or c.fusion_method:
            raise ValueError("TRIP uses workers=0, no AMP and its own fusion")
        if (
            min(
                c.seq_len,
                c.horizon,
                c.batch_size,
                c.num_epochs,
                c.patience,
                c.inference_batch_size,
            )
            < 1
            or c.lr <= 0
        ):
            raise ValueError("Lengths and training settings must be positive")
        if (
            c.lr != 0.001
            or c.batch_size != 128
            or c.num_epochs != 100
            or (c.patience != 10)
        ):
            raise ValueError(
                "Stage 1 requires Adam lr=0.001, batch_size=128, max_epochs=100, patience=10"
            )
        self.device = torch.device(
            c.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.check_point = None
        self.CovariateFusion = None

    @property
    def model_name(self):
        return "TRIP"

    def _init_model(self):
        return TRIPModel(self.config)

    def _init_criterion(self):
        return nn.L1Loss(reduction="none")

    def _process(
        self, input, target=None, input_mark=None, target_mark=None, exog_future=None
    ):
        if exog_future is None:
            raise ValueError("TRIP requires supplied future covariates")
        return {"output": self.model(input, exog_future)}

    def validate(self, valid_data_loader, series_dim=1, criterion=None):
        self.model.eval()
        predictions, targets = ([], [])
        with torch.no_grad():
            for history, labels, future in valid_data_loader:
                prediction = self.model(history.to(self.device), future.to(self.device))
                predictions.append(prediction[:, :, 0].cpu().numpy())
                targets.append(labels[:, :, 0].numpy())
        error = np.concatenate(predictions).astype("float64") - np.concatenate(targets)
        self.val_mse = float((error * error).mean())
        return float(np.abs(error).mean())

    def forecast_fit(
        self, train_valid_data, *, covariates=None, train_ratio_in_tv=1.0, **kwargs
    ):
        c = self.config
        exog = (covariates or {}).get("exog")
        if exog is None or not train_valid_data.index.equals(exog.index):
            raise ValueError("Target and exog must have identical timestamps")
        if (
            not train_valid_data.index.is_unique
            or not train_valid_data.index.is_monotonic_increasing
        ):
            raise ValueError("Expected unique chronological timestamps")
        y = train_valid_data.to_numpy(dtype=np.float64)
        x = exog.to_numpy(dtype=np.float64)
        if y.shape[1] != 1 or x.shape[1] < 1:
            raise ValueError("Expected one target and at least one exogenous channel")
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("Inputs contain NaN or Inf")
        if not 0 < train_ratio_in_tv < 1:
            raise ValueError("TRIP requires a validation split")
        split = int(len(y) * train_ratio_in_tv)
        if split < c.seq_len + c.horizon or len(y) - split < c.horizon:
            raise ValueError("Insufficient complete training/validation windows")
        self.target_columns = list(train_valid_data.columns)
        self.exog_columns = list(exog.columns)
        c.series_dim = 1
        c.input_dim = 1 + x.shape[1]
        c.output_dim = 1
        y = self.scaler1.fit(y[:split]).transform(y).astype(np.float32)
        x = self.scaler2.fit(x[:split]).transform(x).astype(np.float32)
        train_ds = _Windows(y, x, c.seq_len, c.horizon, c.seq_len, split - c.horizon)
        valid_ds = _Windows(y, x, c.seq_len, c.horizon, split, len(y) - c.horizon)
        fix_random_seed(2021)
        generator = torch.Generator().manual_seed(torch.initial_seed())
        train_loader = DataLoader(
            train_ds,
            batch_size=c.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=c.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(2021),
        )
        self.model = self._init_model().to(self.device)
        self.model.adapter_enabled = False
        self.model.controller.requires_grad_(False)
        self.model.routing_weight.requires_grad_(False)
        self.model.decay.requires_grad_(False)
        self.model.target_module.adaptive_local_enabled = False
        for name in ("a_raw", "b_raw", "c", "beta"):
            getattr(self.model.target_module, name).requires_grad_(False)
        self.model.reset_adaptation()
        for _, parameter in self.model.adaptation_parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.001
        )
        self.trainable_params = sum((p.numel() for p in self.model.parameters()))
        print(f"TRIP parameters: {self.trainable_params}", flush=True)
        self.check_point = None
        self.train_history = []
        best, stale = (float("inf"), 0)
        for epoch in range(1, c.num_epochs + 1):
            self.model.train()
            for history, labels, future in train_loader:
                history = history.to(self.device)
                labels = labels.to(self.device)
                future = future.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self._process(history, exog_future=future)["output"]
                scale = torch.sqrt(history[:, :, 0].var(1, unbiased=False) + 1e-05)
                error = (prediction - labels) / scale[:, None, None]
                loss = error.abs().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
            val_mae = self.validate(valid_loader)
            if not np.isfinite([val_mae, self.val_mse]).all():
                raise RuntimeError("Nonfinite validation score")
            self.train_history.append(
                {"epoch": epoch, "val_mae_norm": val_mae, "val_mse_norm": self.val_mse}
            )
            print(
                f"epoch={epoch} val_mae={val_mae:.8f} val_mse={self.val_mse:.8f}",
                flush=True,
            )
            if val_mae < best:
                best, stale = (val_mae, 0)
                self.best_epoch = epoch
                self.check_point = self.save_checkpoint({"Model": self.model})
            else:
                stale += 1
                if stale >= c.patience:
                    break
        self.model.load_state_dict(self.check_point["Model"])
        fit_adaptation(self, train_ds, valid_ds)
        return self

    def batch_forecast(self, horizon, batch_maker, exog_futures, i, **kwargs):
        c = self.config
        if self.model is None:
            raise RuntimeError("Call forecast_fit first")
        if horizon != c.horizon or exog_futures is None:
            raise ValueError("Expected fitted horizon and supplied future exog")
        batch = batch_maker.make_batch(c.inference_batch_size, c.seq_len)
        hy = np.asarray(batch["input"])
        hx = np.asarray(batch["covariates"]["exog"])
        n = len(hy)
        channels = c.input_dim - 1
        if hy.shape != (n, c.seq_len, 1) or hx.shape != (n, c.seq_len, channels):
            raise ValueError("Prediction input shapes do not match fitted model")
        if not (np.isfinite(hy).all() and np.isfinite(hx).all()):
            raise ValueError("Prediction inputs contain NaN or Inf")
        hy = (hy - self.scaler1.mean_) / self.scaler1.scale_
        hx = (hx - self.scaler2.mean_) / self.scaler2.scale_
        history = np.concatenate((hy, hx), axis=2).astype(np.float32)
        start = int(i) * c.inference_batch_size
        fx = np.asarray(exog_futures[start : start + n])
        if fx.shape != (n, horizon, channels):
            raise ValueError("Prediction future exog shape mismatch")
        if not np.isfinite(fx).all():
            raise ValueError("Prediction future exog contains NaN or Inf")
        fx = (fx - self.scaler2.mean_) / self.scaler2.scale_
        fut_tensor = torch.from_numpy(fx.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            prediction = (
                self._process(
                    torch.from_numpy(history).to(self.device), exog_future=fut_tensor
                )["output"]
                .cpu()
                .numpy()
            )
        output = np.empty_like(prediction, dtype=np.float32)
        output[:, :, 0] = (
            prediction[:, :, 0] * self.scaler1.scale_[0] + self.scaler1.mean_[0]
        )
        return output

    def forecast(self, horizon, series, *, covariates=None):
        raise NotImplementedError(
            "Use rolling batch_forecast with supplied future exog"
        )
