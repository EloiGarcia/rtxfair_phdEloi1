from typing import Callable, Optional
import torch

@torch.no_grad()
def fuse_attributions(ig: torch.Tensor, attn: torch.Tensor, beta: float = 0.7) -> torch.Tensor:
    ig_norm = ig / (ig.abs().sum(dim=-1, keepdim=True) + 1e-8)
    attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    return beta * ig_norm + (1.0 - beta) * attn_norm

def integrated_gradients(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 16,
) -> torch.Tensor:
    if baseline is None:
        baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps, device=x.device).view(steps, 1, 1)
    x_interp = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    x_interp.requires_grad_(True)
    y = f(x_interp.view(-1, x.shape[-1])).view(steps, x.shape[0], 1)
    grads = []
    for s in range(steps):
        if x_interp.grad is not None:
            x_interp.grad = None
        y[s].sum().backward(retain_graph=True)
        grads.append(x_interp.grad[s].detach().clone())
    grads = torch.stack(grads, dim=0)
    avg = grads.mean(dim=0)
    ig = (x - baseline) * avg
    return ig

@torch.no_grad()
def infidelity(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, E: torch.Tensor, sigma: float = 0.1, n: int = 50) -> torch.Tensor:
    """Yeh et al. infidelity estimate with Gaussian perturbations. Returns (B,) per-sample."""
    B, d = x.shape
    u = torch.randn(n, B, d, device=x.device) * sigma
    fx = f(x)
    diffs = ((u * E).sum(dim=-1, keepdim=True) - (fx - f(x - u))) ** 2
    return diffs.mean(dim=0).view(B)

@torch.no_grad()
def stability(E_fn, x: torch.Tensor, sigma: float = 0.05, n: int = 20) -> torch.Tensor:
    """Mean cosine similarity of E(x) under Gaussian noise; higher is more stable. Returns (B,)."""
    B, d = x.shape
    E0 = E_fn(x)
    sims = []
    for _ in range(n):
        xp = x + torch.randn_like(x) * sigma
        Ep = E_fn(xp)
        num = (E0 * Ep).sum(dim=-1)
        den = (E0.norm(dim=-1) * Ep.norm(dim=-1) + 1e-8)
        sims.append(num / den)
    return torch.stack(sims, dim=0).mean(dim=0)
