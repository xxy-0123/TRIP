"""Stage 2 fitting of the supplied-future adaptation parameters."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def _bounds(name):
    if name.endswith(".beta") or name == "decay":
        return (0.0, 1.0)
    if name == "routing_weight":
        return (0.0, 2.0)
    return (None, None)


@torch.no_grad()
def _cache(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        generator=torch.Generator().manual_seed(2021),
    )
    out = {}
    for history, labels, future in loader:
        history, future = (history.to(device), future.to(device))
        d = model.encode(history, future)
        features = {
            key: d[key].detach()
            for key in ("ef", "eh", "last", "mu", "s", "es", "global_part")
        }
        yn = (history[:, :, 0] - d["mu"]) / d["s"]
        previous = torch.diff(yn[:, -25:-1], dim=1)
        delta = yn[:, -1:] - yn[:, -2:-1]
        scale = previous.abs().mean(1, keepdim=True) + 1e-08
        features.update(
            recent=yn[:, -24:],
            stable=yn[:, -25:-1].mean(1, keepdim=True),
            shock=torch.log1p(delta.abs() / scale).clamp(0, 12),
            alignment=(
                delta * previous.mean(1, keepdim=True) / (delta.abs() * scale + 1e-08)
            ).clamp(-1, 1),
            y=labels[:, :, 0].to(device),
        )
        for key, value in features.items():
            out.setdefault(key, []).append(value)
    return {key: torch.cat(values) for key, values in out.items()}


def _forward(model, cache):
    tm = model.target_module
    q = (
        torch.nn.functional.softplus(tm.a_raw) * cache["shock"]
        - torch.nn.functional.softplus(tm.b_raw) * cache["alignment"]
        + tm.c
    )
    rho = 1 - tm.beta * torch.sigmoid(q)
    reference = rho * cache["last"] + (1 - rho) * cache["stable"]
    loc = tm.local_proj(cache["recent"] - reference)
    return model.decode(dict(cache, loc=loc, rho=rho))


def _slice(cache, indices):
    return {key: value[indices] for key, value in cache.items()}


def _objective(prediction, cache):
    error = (prediction - cache["y"]) / cache["s"]
    return (
        error.square().add(1e-06).sqrt() - 0.001
    ).mean() + 0.1 * error.square().mean()


@torch.no_grad()
def _metrics(model, cache):
    total_abs = total_sq = 0.0
    count = 0
    for indices in torch.arange(len(cache["y"]), device=cache["y"].device).split(128):
        chunk = _slice(cache, indices)
        error = _forward(model, chunk).double() - chunk["y"].double()
        total_abs += error.abs().sum().item()
        total_sq += error.square().sum().item()
        count += error.numel()
    return {"mae": total_abs / count, "mse": total_sq / count}


def fit_adaptation(owner, train_ds, valid_ds):
    """Freeze the selected backbone and restore the best validation checkpoint."""
    import scipy
    from scipy.optimize import minimize

    if scipy.__version__ != "1.10.1":
        raise RuntimeError("Validated Stage 2 requires SciPy 1.10.1")
    model, device = (owner.model, owner.device)
    if not len(train_ds) or not len(valid_ds):
        raise ValueError("Stage 2 requires nonempty training and validation windows")
    owner.backbone_best_epoch = owner.best_epoch
    owner.backbone_train_history = owner.train_history
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    model.eval()
    model.reset_adaptation()
    model.adapter_enabled = True
    model.uniform_persistence = False
    model.target_module.adaptive_local_enabled = True
    parameters = model.adaptation_parameters()
    for _, parameter in parameters:
        parameter.requires_grad_(True)
    if sum((p.numel() for _, p in parameters)) != 2 * model.exog_dim + 11:
        raise RuntimeError("Unexpected Stage 2 adaptation parameter count")
    train = _cache(model, train_ds, device)
    valid = _cache(model, valid_ds, device)
    best, state, selected = (float("inf"), None, 0)
    history = []
    calls = iterations = 0

    def validate(index):
        nonlocal best, state, selected
        metrics = _metrics(model, valid)
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError("Nonfinite Stage 2 validation metric")
        history.append(dict(index=index, **metrics))
        if metrics["mae"] < best:
            best, selected = (metrics["mae"], index)
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    validate(0)
    scales, boxes = ([], [])
    for name, parameter in parameters:
        lo, hi = _bounds(name)
        scale = (
            hi - lo
            if lo is not None
            else max(1.0, parameter.detach().abs().max().item())
        )
        scales.extend([scale] * parameter.numel())
        boxes.extend(
            [(None if lo is None else lo / scale, None if hi is None else hi / scale)]
            * parameter.numel()
        )
    scales = np.array(scales, dtype=np.float64)
    x0 = (
        np.concatenate(
            [
                parameter.detach().cpu().numpy().reshape(-1)
                for _, parameter in parameters
            ]
        ).astype(np.float64)
        / scales
    )

    def assign(coordinates):
        offset = 0
        with torch.no_grad():
            for _, parameter in parameters:
                size = parameter.numel()
                parameter.copy_(
                    torch.as_tensor(
                        (coordinates * scales)[offset : offset + size].reshape(
                            parameter.shape
                        ),
                        device=device,
                        dtype=parameter.dtype,
                    )
                )
                offset += size

    class BudgetStop(Exception):
        pass

    def closure(coordinates):
        nonlocal calls
        if calls >= 60:
            raise BudgetStop()
        calls += 1
        assign(coordinates)
        model.zero_grad(set_to_none=True)
        total = 0.0
        count = train["y"].numel()
        for indices in torch.arange(len(train["y"]), device=device).split(1024):
            chunk = _slice(train, indices)
            loss = _objective(_forward(model, chunk), chunk) * (
                chunk["y"].numel() / count
            )
            loss.backward()
            total += loss.item()
        gradient = (
            np.concatenate(
                [
                    parameter.grad.detach().cpu().numpy().reshape(-1)
                    for _, parameter in parameters
                ]
            ).astype(np.float64)
            * scales
        )
        if not np.isfinite(total) or not np.isfinite(gradient).all():
            raise RuntimeError("Nonfinite Stage 2 objective or gradient")
        return (total, gradient)

    def callback(coordinates):
        nonlocal iterations
        assign(coordinates)
        iterations += 1
        validate(iterations)

    try:
        f0, g0 = closure(x0)
        f1, g1 = closure(x0)
        if f0 != f1 or not np.array_equal(g0, g1):
            raise RuntimeError("Non-deterministic Stage 2 closure")
        result = minimize(
            closure,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=boxes,
            callback=callback,
            options=dict(maxiter=30, maxfun=60, maxls=8),
        )
        status = dict(
            message=str(result.message),
            success=bool(result.success),
            nit=int(result.nit),
            nfev=int(result.nfev),
        )
    except BudgetStop:
        status = dict(
            message="Hard total closure budget 60 reached; keep best accepted validation checkpoint",
            success=False,
        )
    finally:
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        model.eval()
    owner.check_point = {"Model": state}
    owner.adapter_history = {"adaptation": history}
    owner.adapter_best_epochs = {"adaptation": selected}
    owner.adapter_solver_status = status
    owner.adapter_closure_calls = calls
    owner.adapter_iterations = iterations
    owner.best_epoch = selected
    owner.val_mse = history[selected]["mse"]
    owner.trainable_params = sum((p.numel() for _, p in parameters))
    print(
        f"TRIP adaptation: selected iteration={selected} validation MAE={best:.8f} MSE={owner.val_mse:.8f} closures={calls}",
        flush=True,
    )
