from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from .model import TabularAttentionNet

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

@dataclass
class TrainConfig:
    hidden: int = 64
    dropout: float = 0.0
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"

def train_model(X_train, y_train, X_val, y_val, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    d_in = X_train.shape[1]
    model = TabularAttentionNet(d_in=d_in, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf"); best_state = None
    for epoch in range(cfg.epochs):
        model.train(); n=0; loss_sum=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat, _ = model(xb)
            loss = F.binary_cross_entropy(yhat, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); n += len(xb)
        # val
        model.eval(); vm=0; vloss=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat, _ = model(xb)
                vloss += F.binary_cross_entropy(yhat, yb).item() * len(xb); vm += len(xb)
        vloss /= vm
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{cfg.epochs} - train_bce={loss_sum/n:.4f} - val_bce={vloss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
