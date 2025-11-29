import argparse, json
from sklearn.model_selection import train_test_split
from rtxfair.data import load_heloc_csv
from rtxfair.train import train_model, TrainConfig
from rtxfair.metrics import summary_metrics
import torch, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    X, y = load_heloc_csv(args.csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    cfg = TrainConfig(hidden=args.hidden, epochs=args.epochs, device=args.device)
    model = train_model(Xtr, ytr, Xte, yte, cfg)
    metrics = summary_metrics(model, Xte, yte, device=args.device)
    print(json.dumps(metrics, indent=2))

    os.makedirs("./artifacts", exist_ok=True)
    torch.save(model.state_dict(), "./artifacts/rtxfair_weights.pt")
    print("Saved weights to ./artifacts/rtxfair_weights.pt")

if __name__ == "__main__":
    main()
