"""CUDA-default Adam / AdamW optimizers (+ legacy DirectML-safe variants).

History: stock ``torch.optim.Adam`` updates moments via ``aten::lerp``,
which had no DirectML kernel, so every step round-tripped tensors to CPU.
``DMLAdam``/``DMLAdamW`` spelled the same math with DML-native ops
(``mul_``/``add_``/``sqrt``/``div``) for the old WSL2 + RX 6900 XT setup.

On CUDA (Google Colab) stock Adam/AdamW have native foreach/fused kernels
and are faster, so :func:`build_optimizer_for_device` now returns stock
torch optimizers on CUDA/CPU. The ``DML*`` classes stay importable as a
legacy resume shim (old ``last.pt`` files store their state) but are no
longer the default path.

Usage (sentinel / overlord)::

    from dml_optimizer import build_optimizer_for_device
    self.optimizer = build_optimizer_for_device(
        self.device, self.q_net.parameters(), lr=LR)

On CUDA/CPU the factory returns stock ``torch.optim.Adam`` / ``AdamW``
(foreach where available). ``name='lion'`` returns the sign-momentum Lion
implementation (CUDA-native ops only) on all devices.
"""

import math

import torch
from torch.optim import Optimizer


def is_cuda_device(device) -> bool:
    """True if ``device`` is a CUDA device."""
    try:
        t = getattr(device, "type", device)
        return str(t).lower() == "cuda"
    except Exception:
        return False


def is_dml_device(device) -> bool:
    """Legacy shim: True only for old DirectML ``privateuse`` devices.

    DirectML is removed (CUDA default). Kept so old checkpoints/logs that
    reference ``privateuseone`` still classify correctly; always False on
    Colab (cuda/cpu only).
    """
    try:
        t = getattr(device, "type", device)
        return str(t).lower().startswith("privateuse")
    except Exception:
        return False


class DMLAdam(Optimizer):
    """Legacy DirectML-safe Adam (kept for old-checkpoint resume).

    Mathematically identical to Adam; only the op spelling differs to avoid
    ``aten::lerp``. Uses only ``mul_``/``add_``/``sqrt``/``div``.
    New CUDA/CPU runs use stock ``torch.optim.Adam`` via the factory.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        *,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        if amsgrad:
            raise NotImplementedError("DMLAdam does not support amsgrad (no DML kernel)")
        if maximize:
            raise NotImplementedError("DMLAdam does not support maximize=True")
        if capturable:
            raise NotImplementedError("DMLAdam does not support capturable=True")
        if differentiable:
            raise NotImplementedError("DMLAdam does not support differentiable=True")
        if foreach:
            raise NotImplementedError("DMLAdam does not support foreach=True (no DML lerp kernel)")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DMLAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # Accept both int steps (ours) and Tensor steps (torch.Adam ckpts).
                step_val = state["step"]
                if torch.is_tensor(step_val):
                    step_val = int(step_val.item())
                step_val += 1
                # Keep stored type stable: int for fresh DML state, Tensor stays Tensor.
                if torch.is_tensor(state["step"]):
                    state["step"] = torch.tensor(
                        step_val, dtype=state["step"].dtype, device=state["step"].device
                    )
                else:
                    state["step"] = step_val
                step = step_val

                # Fused L2 decay, matching torch.optim.Adam semantics.
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # DML-safe moment updates (no lerp_, no addcmul_).
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                # denom = sqrt(exp_avg_sq / bias_correction2) + eps, DML-native ops only.
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                # p -= step_size * exp_avg / denom  (no addcdiv_ -> no 2nd fallback).
                p.add_(exp_avg / denom, alpha=-step_size)

        return loss


class DMLAdamW(Optimizer):
    """Legacy DirectML-safe AdamW (kept for old-checkpoint resume)."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        if amsgrad:
            raise NotImplementedError("DMLAdamW does not support amsgrad")
        if maximize:
            raise NotImplementedError("DMLAdamW does not support maximize=True")
        if capturable:
            raise NotImplementedError("DMLAdamW does not support capturable=True")
        if differentiable:
            raise NotImplementedError("DMLAdamW does not support differentiable=True")
        if foreach:
            raise NotImplementedError("DMLAdamW does not support foreach=True")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DMLAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                step_val = state["step"]
                if torch.is_tensor(step_val):
                    step_val = int(step_val.item())
                step_val += 1
                if torch.is_tensor(state["step"]):
                    state["step"] = torch.tensor(
                        step_val, dtype=state["step"].dtype, device=state["step"].device
                    )
                else:
                    state["step"] = step_val
                step = step_val

                # Decoupled decay, matching torch.optim.AdamW semantics.
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                p.add_(exp_avg / denom, alpha=-step_size)

        return loss


class DMLLion(Optimizer):
    """Lion (EvoLved Sign Momentum, Chen et al. 2023) — default Lion on all devices.

    Update: m = beta1*m + (1-beta1)*g; p -= lr * (sign(m) + wd*p).
    One state tensor per param (vs two for Adam) and only
    ``mul_``/``add_``/``sign`` elementwise ops — CUDA/CPU-native, fewer
    dispatches per step than Adam on small nets where launch overhead
    dominates (profiled: optimizer step was ~50% of a 256-batch MLP update).
    Note: Lion LRs run ~3-10x below Adam's; sweep {1e-4, 3e-4, 7e-4}.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DMLLion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avg = state["exp_avg"]
                step_val = state["step"]
                if torch.is_tensor(step_val):
                    step_val = int(step_val.item())
                state["step"] = step_val + 1

                # Decoupled decay, then momentum, then sign update.
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

        return loss


class Lookahead:
    """Lookahead wrapper (Zhang et al. 2019) with lerp-free slow sync.

    Keeps slow weights; every ``k`` fast steps pulls slow toward fast and
    copies back. Sync uses ``add_`` only (DML-native). Costs one extra
    param-sized buffer set; sync amortized over k steps.
    State dict = base optimizer's + slow weights (missing slow weights on
    load -> re-init from current params, with no error).
    """

    def __init__(self, base_optimizer, alpha=0.5, k=5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        self.optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self._step = 0
        self.slow = []
        for group in base_optimizer.param_groups:
            self.slow.append([p.detach().clone() for p in group["params"]])

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    @torch.no_grad()
    def _sync(self):
        for group, slow_group in zip(self.optimizer.param_groups, self.slow):
            for p, q in zip(group["params"], slow_group):
                q.add_(p - q, alpha=self.alpha)
                p.copy_(q)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            self._sync()
        return loss

    def state_dict(self):
        return {"base": self.optimizer.state_dict(), "slow": self.slow,
                "step": self._step, "alpha": self.alpha, "k": self.k}

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd["base"])
        self._step = int(sd.get("step", 0))
        self.alpha = float(sd.get("alpha", self.alpha))
        self.k = int(sd.get("k", self.k))
        saved_slow = sd.get("slow")
        if saved_slow is not None:
            try:
                for group, slow_group, saved_group in zip(
                        self.optimizer.param_groups, self.slow, saved_slow):
                    for p, q, s in zip(group["params"], slow_group, saved_group):
                        q.copy_(s.to(q.device))
                return
            except Exception:
                pass
        for group, slow_group in zip(self.optimizer.param_groups, self.slow):
            for p, q in zip(group["params"], slow_group):
                q.copy_(p.detach())


def schedule_factor(step, warmup=5000, total=200000, min_ratio=0.1):
    """Pure-function LR schedule: linear warmup then cosine decay.

    Stateless in the checkpoint sense: callers compute
    ``lr = base_lr * schedule_factor(total_steps)`` each update, so resume
    needs no extra state (total_steps is already checkpointed).
    """
    if step < 0:
        step = 0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    import math
    if step >= total:
        return min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_optimizer_for_device(
    device, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, adamw=False,
    name="adam", lookahead=None, tuned=False,
):
    """Return the default optimizer for ``device`` (CUDA-default).

    - CUDA / CPU -> stock ``torch.optim.Adam`` / ``torch.optim.AdamW``
      (foreach/fused where available — fastest on Colab GPUs).
    - Legacy DirectML ``privateuse`` device -> ``DMLAdam``/``DMLAdamW``
      (kept only so ancient local runs still import; never happens on Colab).

    ``name='lion'`` selects :class:`DMLLion` (sign-momentum Lion, CUDA-native
    ``mul_``/``add_``/``sign`` ops only — torch has no stock Lion).
    ``tuned=True`` applies the DQN task preset (eps=1e-4, decay=1e-4) to Adam.
    ``lookahead={'alpha': 0.5, 'k': 5}`` wraps the base optimizer.
    Defaults preserve the pre-existing behavior exactly.
    """
    import torch.optim as optim

    if tuned and name == "adam" and eps == 1e-8:
        eps = 1e-4
        if weight_decay == 0.0:
            weight_decay = 1e-4
    if name == "lion":
        # Lion has no stock torch equivalent; ours is CUDA/CPU-native.
        base = DMLLion(params, lr=lr, betas=betas[:2], weight_decay=weight_decay)
    elif is_dml_device(device):
        # Legacy DirectML path only (never on Colab).
        cls = DMLAdamW if adamw else DMLAdam
        base = cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        cls = optim.AdamW if adamw else optim.Adam
        try:
            # foreach=True is the fastest CUDA kernel; falls back cleanly on CPU.
            base = cls(params, lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, foreach=True)
        except Exception:
            base = cls(params, lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay)
    if lookahead:
        alpha = float(lookahead.get("alpha", 0.5))
        k = int(lookahead.get("k", 5))
        return Lookahead(base, alpha=alpha, k=k)
    return base


def assert_params_on_device(model, device, what="model"):
    """Raise with a clear message if any param is not on ``device``.

    Used by training setup + verification to prove GPU utilization instead of
    silently falling back to CPU.
    """
    want = getattr(device, "type", str(device))
    for name, par in model.named_parameters():
        got = getattr(par.device, "type", str(par.device))
        if str(par.device) != str(device) and got != want:
            # Compare full device strings (cuda:0 vs cpu) for clarity.
            raise AssertionError(
                f"{what} param {name} on {par.device}, expected {device} "
                f"(model silently on wrong device -> CPU training)"
            )
    return True


__all__ = [
    "DMLAdam",
    "DMLAdamW",
    "DMLLion",
    "Lookahead",
    "schedule_factor",
    "build_optimizer_for_device",
    "is_dml_device",
    "is_cuda_device",
    "assert_params_on_device",
]
