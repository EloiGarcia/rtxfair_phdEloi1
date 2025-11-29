import argparse, os, torch
from rtxfair.models import TabularAttentionNet
from rtxfair.export import export_torchscript

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="./artifacts/model.pt")
    args = ap.parse_args()

    m = TabularAttentionNet(d_in=args.d).eval()
    if args.weights and os.path.exists(args.weights):
        m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    path = export_torchscript(m, d_in=args.d, out_path=args.out)
    print("Saved TorchScript to", path)

if __name__ == "__main__":
    main()
