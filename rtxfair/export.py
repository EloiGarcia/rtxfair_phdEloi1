import os, torch

def export_torchscript(model, d_in: int, out_path: str):
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x: torch.Tensor):
            pd, attn = self.m(x)
            return pd, attn

    wrapper = InferenceWrapper(model).eval()
    example = torch.randn(1, d_in)
    ts = torch.jit.trace(wrapper, example)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts.save(out_path)
    return out_path
