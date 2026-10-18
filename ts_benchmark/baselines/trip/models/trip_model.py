"""TRIP forecasting network."""
import math
import torch
from torch import nn


class EndogenousTargetModule(nn.Module):
    def __init__(self, seq_len, horizon):
        super().__init__()
        if seq_len < 25:
            raise ValueError("TRIP requires seq_len >= 25")
        self.local_len = min(24, seq_len)
        self.global_projection = nn.Linear(seq_len, horizon)
        self.local_proj = nn.Linear(self.local_len, horizon)
        nn.init.zeros_(self.local_proj.weight)
        nn.init.zeros_(self.local_proj.bias)
        self.a_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.b_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.c = nn.Parameter(torch.tensor(-2.0))
        self.beta = nn.Parameter(torch.tensor(0.01))
        self.adaptive_local_enabled = True

    def forward(self, y_norm):
        y_last = y_norm[:, -1:]
        rho = torch.ones_like(y_last)
        y_global = self.global_projection(y_norm)
        y_recent = y_norm[:, -self.local_len :]
        reference = y_last
        if self.adaptive_local_enabled:
            previous = torch.diff(y_norm[:, -25:-1], dim=1)
            delta = y_norm[:, -1:] - y_norm[:, -2:-1]
            scale = previous.abs().mean(1, keepdim=True) + 1e-08
            shock = torch.log1p(delta.abs() / scale).clamp(0, 12)
            alignment = (
                delta * previous.mean(1, keepdim=True) / (delta.abs() * scale + 1e-08)
            ).clamp(-1, 1)
            q = (
                nn.functional.softplus(self.a_raw) * shock
                - nn.functional.softplus(self.b_raw) * alignment
                + self.c
            )
            rho = 1 - self.beta * torch.sigmoid(q)
            stable = y_norm[:, -25:-1].mean(1, keepdim=True)
            reference = rho * y_last + (1 - rho) * stable
        y_local = self.local_proj(y_recent - reference)
        self.cached_global = y_global
        self.cached_local = y_local
        self.cached_rho = rho
        return y_global + self.local_gate(rho) * y_local

    def local_gate(self, rho):
        a = self.local_gate_coeff
        t = self.gate_lead[None, :]
        r = 2 * rho - 1
        z = a[0] + a[1] * r + a[2] * t + a[3] * r * t
        return 2 * torch.sigmoid(z)


class TRIPModel(nn.Module):
    """Input [B,L,1+C], future X [B,H,C]; output [B,H,1]."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.horizon = int(config.pred_len)
        self.exog_dim = int(config.input_dim) - 1
        if min(self.seq_len, self.horizon, self.exog_dim) < 1:
            raise ValueError("L, H and C must be positive")
        self.target_module = EndogenousTargetModule(self.seq_len, self.horizon)
        self.response_net = nn.Sequential(
            nn.Linear(self.exog_dim, 16), nn.GELU(), nn.Linear(16, 1, bias=False)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        rank = min(8, self.seq_len, self.horizon)
        self.exo_u = nn.Linear(self.seq_len, rank, bias=False)
        self.exo_v = nn.Linear(rank, self.horizon, bias=True)
        self.combiner = nn.Sequential(nn.Linear(7, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.combiner[2].weight)
        nn.init.zeros_(self.combiner[2].bias)
        self.register_buffer(
            "lead",
            torch.arange(1, self.horizon + 1, dtype=torch.float32) / self.horizon,
            persistent=False,
        )
        self.controller = nn.Linear(2 * self.exog_dim, 1)
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)
        self.routing_weight = nn.Parameter(torch.tensor(0.0))
        self.decay = nn.Parameter(torch.tensor(1.0))
        self.adapter_enabled = True
        self.uniform_persistence = False
        tm = self.target_module
        tm.local_gate_coeff = nn.Parameter(torch.zeros(4))
        tm.register_buffer(
            "gate_lead", torch.linspace(-1, 1, self.horizon), persistent=False
        )
        self.reset_adaptation()

    def adaptation_parameters(self):
        names = (
            "routing_weight",
            "decay",
            "target_module.a_raw",
            "target_module.b_raw",
            "target_module.c",
            "target_module.beta",
            "target_module.local_gate_coeff",
            "controller.weight",
            "controller.bias",
        )
        parameters = dict(self.named_parameters())
        return [(name, parameters[name]) for name in names]

    @torch.no_grad()
    def reset_adaptation(self):
        tm = self.target_module
        tm.a_raw.fill_(math.log(math.expm1(1.0)))
        tm.b_raw.fill_(math.log(math.expm1(1.0)))
        tm.c.fill_(-2)
        tm.beta.zero_()
        tm.local_gate_coeff.zero_()
        self.controller.weight.zero_()
        self.controller.bias.zero_()
        self.routing_weight.zero_()
        self.decay.fill_(1)

    def forward(self, history, future):
        if self.adapter_enabled:
            return self.decode(self.encode(history, future)).unsqueeze(-1)
        return self.forward_backbone(history, future)

    def encode(self, history, future):
        """Encode supplied-future features, including current target adaptation."""
        n = len(history)
        if history.shape != (n, self.seq_len, 1 + self.exog_dim):
            raise ValueError("Expected history [B,L,1+C]")
        if future is None or future.shape != (n, self.horizon, self.exog_dim):
            raise ValueError("Expected supplied future exog [B,H,C]")
        y, x = (history[:, :, 0], history[:, :, 1:])
        mu = y.mean(1, keepdim=True)
        s = (y.var(1, unbiased=False, keepdim=True) + 1e-05).sqrt()
        yn = (y - mu) / s
        xm = x.mean(1, keepdim=True)
        xs = (x.var(1, unbiased=False, keepdim=True) + 1e-05).sqrt()
        xf = (future - xm) / xs
        xx = torch.cat([(x - xm) / xs, xf], 1)
        g = 3 * torch.tanh(
            (self.response_net(xx) - self.response_net(torch.zeros_like(xx[:1, :1])))
            / 3
        ).squeeze(-1)
        es = torch.cat([xf.mean(1), (future.mean(1) - x[:, -1]) / xs[:, 0]], 1)
        es = torch.sign(es) * torch.log1p(es.abs())
        tm = self.target_module
        en = tm(yn)
        d = dict(
            en=en,
            ef=self.gamma * g[:, self.seq_len :],
            eh=self.exo_v(self.exo_u(g[:, : self.seq_len])),
            last=yn[:, -1:],
            mu=mu,
            s=s,
            es=torch.tanh(es / 3),
        )
        d.update(loc=tm.cached_local, rho=tm.cached_rho, global_part=tm.cached_global)
        return d

    def decode(self, d):
        d = dict(
            d, en=d["global_part"] + self.target_module.local_gate(d["rho"]) * d["loc"]
        )
        alpha = torch.exp(math.log(4) * torch.tanh(self.controller(d["es"])))
        instant = alpha * d["ef"]
        if self.uniform_persistence:
            persist = (
                instant.cumsum(1)
                / torch.arange(1, self.horizon + 1, device=instant.device)[None, :]
            )
        else:
            lam = self.decay.clamp(0, 1)
            idx = torch.arange(self.horizon, device=instant.device)
            lag = idx[:, None] - idx[None, :]
            w = lam.pow(lag.clamp(min=0)) * (lag >= 0)
            w = w / w.sum(1, keepdim=True)
            persist = instant @ w.T
        routed = instant + self.routing_weight.clamp(0, 2) * persist
        en, n = (d["en"], len(instant))
        z = torch.stack(
            [
                en,
                d["last"].expand(-1, self.horizon),
                routed,
                d["eh"],
                d["mu"].expand(-1, self.horizon),
                d["s"].expand(-1, self.horizon),
                self.lead[None, :].expand(n, -1),
            ],
            -1,
        )
        return (en + self.combiner(z).squeeze(-1)) * d["s"] + d["mu"]

    def forward_backbone(self, history, future):
        n = len(history)
        if history.shape != (n, self.seq_len, 1 + self.exog_dim):
            raise ValueError("Expected history [B,L,1+C]")
        if future is None or future.shape != (n, self.horizon, self.exog_dim):
            raise ValueError("Expected supplied future exog [B,H,C]")
        y_raw = history[:, :, 0]
        x_raw = history[:, :, 1:]
        mean_y = y_raw.mean(dim=1, keepdim=True)
        scale_y = torch.sqrt(y_raw.var(dim=1, unbiased=False, keepdim=True) + 1e-05)
        y_norm = (y_raw - mean_y) / scale_y
        mean_x = x_raw.mean(dim=1, keepdim=True)
        scale_x = torch.sqrt(x_raw.var(dim=1, unbiased=False, keepdim=True) + 1e-05)
        xh_norm = (x_raw - mean_x) / scale_x
        xf_norm = (future - mean_x) / scale_x
        x_all = torch.cat((xh_norm, xf_norm), dim=1)
        y_last = y_norm[:, -1:]
        y_endo = self.target_module(y_norm)
        zero_anchor = torch.zeros(
            1, 1, self.exog_dim, device=history.device, dtype=history.dtype
        )
        g_raw = self.response_net(x_all) - self.response_net(zero_anchor)
        g_bounded = 3.0 * torch.tanh(g_raw / 3.0).squeeze(-1)
        gh = g_bounded[:, : self.seq_len]
        gf = g_bounded[:, self.seq_len :]
        exo_f_out = self.gamma * gf
        exo_h_out = self.exo_v(self.exo_u(gh))
        y_last_exp = y_last[:, :, None].expand(-1, self.horizon, 1)
        target_stats = torch.cat((mean_y, scale_y), dim=-1)
        target_stats = target_stats[:, None, :].expand(-1, self.horizon, -1)
        lead_exp = self.lead[None, :, None].expand(n, -1, -1)
        combiner_inputs = torch.cat(
            [
                y_endo[:, :, None],
                y_last_exp,
                exo_f_out[:, :, None],
                exo_h_out[:, :, None],
                target_stats,
                lead_exp,
            ],
            dim=-1,
        )
        delta = self.combiner(combiner_inputs).squeeze(-1)
        y_norm_pred = y_endo + delta
        return (y_norm_pred * scale_y + mean_y).unsqueeze(-1)
