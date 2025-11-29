import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import torch

def calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < bins[i+1])
        if m.any():
            e = np.abs(prob[m].mean() - y[m].mean())
            ece += e * m.mean()
    return float(ece)

@torch.no_grad()
def predict_prob(model, X, device="cpu", batch_size=1024):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(X.values, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    outs = []
    for (xb,) in dl:
        xb = xb.to(device)
        y, _ = model(xb)
        outs.append(y.cpu().numpy())
    return np.vstack(outs).ravel()

def summary_metrics(model, X_test, y_test, device="cpu"):
    prob = predict_prob(model, X_test, device=device)
    auc = roc_auc_score(y_test, prob)
    brier = brier_score_loss(y_test, prob)
    ece = calibration_error(prob, y_test)
    return {"AUC": float(auc), "Brier": float(brier), "ECE": float(ece)}
