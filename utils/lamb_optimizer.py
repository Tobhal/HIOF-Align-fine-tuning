import math
import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """
    LAMB optimizer with optional AdamW-style (decoupled) weight decay.
    - If decouple_weight_decay=False: weight decay is applied to adam_step (coupled, L2-style)
    - If decouple_weight_decay=True: weight decay is applied directly to params (AdamW-style)

    fixed_decay:
      - If True:    p *= (1 - weight_decay)          (NOT scaled by lr)
      - If False:   p *= (1 - lr * weight_decay)     (AdamW common form)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        adam=False,
        maximize=False,
        decouple_weight_decay: bool = False,
        fixed_decay: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.adam = adam
        self.maximize = maximize
        self.decouple_weight_decay = decouple_weight_decay
        self.fixed_decay = fixed_decay

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data if not self.maximize else -p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # denom
                denom = exp_avg_sq.sqrt().add_(eps)

                # "adam step" (WITHOUT decoupled weight decay)
                adam_step = exp_avg / denom

                # Coupled (L2-style) weight decay (old behavior)
                if (wd != 0) and (not self.decouple_weight_decay):
                    adam_step.add_(p.data, alpha=wd)

                # Trust ratio uses the step direction (typically excluding decoupled WD)
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                adam_norm = adam_step.pow(2).sum().sqrt()

                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = (weight_norm / adam_norm).item()

                if self.adam:
                    trust_ratio = 1.0

                # AdamW-style decoupled weight decay:
                # apply decay directly to weights, separate from adam_step
                if (wd != 0) and self.decouple_weight_decay:
                    if self.fixed_decay:
                        # p *= (1 - wd)
                        p.data.mul_(1.0 - wd)
                    else:
                        # p *= (1 - lr * wd)
                        p.data.mul_(1.0 - lr * wd)

                # parameter update
                p.data.add_(adam_step, alpha=-lr * trust_ratio)

                # for logging / debugging if you want
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio

        return loss