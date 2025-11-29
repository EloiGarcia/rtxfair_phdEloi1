import torch
from rtxfair.models import TabularAttentionNet
from rtxfair.explain import integrated_gradients, fuse_attributions

def test_forward_and_explain():
    d = 10
    m = TabularAttentionNet(d_in=d).eval()
    x = torch.randn(2, d)
    with torch.no_grad():
        pd, attn = m(x)
    assert pd.shape == (2,1) and attn.shape == (2,d)
    def f(z):
        out, _ = m(z)
        return out
    ig = integrated_gradients(f, x, steps=8)
    E  = fuse_attributions(ig, attn, beta=0.7)
    assert E.shape == (2,d)
