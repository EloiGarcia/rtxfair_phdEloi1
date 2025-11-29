# rtxfair/latency.py
import time, csv
import numpy as np
import torch
from .explain import integrated_gradients, fuse_attributions

def _to_tensor_rows(X, device):
    """Devuelve un callable idx->tensor [1,d] a partir de DataFrame o Tensor."""
    if isinstance(X, torch.Tensor):
        d = X.shape[1]
        def get(i): 
            return X[i:i+1].to(device)
        return d, get
    else:
        # asumimos pandas.DataFrame
        d = X.shape[1]
        def get(i):
            x = torch.tensor(X.iloc[i:i+1].values, dtype=torch.float32, device=device)
            return x
        return d, get

def benchmark_latency(model, X, steps=16, n=200, device="cpu", out_csv=None):
    device = torch.device(device)
    model = model.to(device).eval()
    d, get_row = _to_tensor_rows(X, device)

    pred_ms, exp_ms = [], []

    fh = None
    writer = None
    if out_csv:
        fh = open(out_csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["idx","pred_ms","exp_ms","pd"])

    # warmup
    _ = model(torch.randn(1, d, device=device))

    for i in range(min(n, len(X))):
        x = get_row(i)

        # pred timing
        t0 = time.perf_counter()
        pd, attn = model(x)
        t1 = time.perf_counter()
        pred_ms.append((t1 - t0) * 1000.0)

        # IG + fusion timing
        def pred_fn(z):
            y, _ = model(z)
            return y

        t2 = time.perf_counter()
        ig = integrated_gradients(pred_fn, x, baseline=torch.zeros_like(x), steps=steps)
        E  = fuse_attributions(ig, attn, beta=0.7)
        _ = float(E.sum())  # evita lazy eval
        t3 = time.perf_counter()
        exp_ms.append((t3 - t2) * 1000.0)

        if writer:
            writer.writerow([i, f"{pred_ms[-1]:.3f}", f"{exp_ms[-1]:.3f}", f"{float(pd.item()):.6f}"])

    if fh:
        fh.close()

    def stats(a):
        a = np.array(a)
        return {"mean": float(a.mean()), "p90": float(np.quantile(a, 0.9)), "p99": float(np.quantile(a, 0.99))}
    return {"pred_ms": stats(pred_ms), "exp_ms": stats(exp_ms)}
