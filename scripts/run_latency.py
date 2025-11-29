import argparse, json
from sklearn.model_selection import train_test_split
from rtxfair.data import load_heloc_csv
from rtxfair.train import train_model, TrainConfig
from rtxfair.latency import benchmark_latency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    X, y = load_heloc_csv(args.csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    cfg = TrainConfig(epochs=5, device=args.device)
    model = train_model(Xtr, ytr, Xte, yte, cfg)

    stats = benchmark_latency(model, Xte, steps=args.steps, n=args.n, device=args.device, out_csv="latency_log.csv")
    print(json.dumps(stats, indent=2))
    print("Wrote latency_log.csv")

if __name__ == "__main__":
    main()
