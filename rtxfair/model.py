import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularAttentionNet(nn.Module):
    """Feature-wise attention + MLP head.
    forward(x) -> (pd, attn) where pd in [0,1], attn sums to 1 across features.
    """
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.attn_w = nn.Linear(d_in, d_in, bias=True)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.do = nn.Dropout(dropout)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.attn_w(x)
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor):
        attn = self.attention(x)          # [B,d]
        x_att = attn * x                  # [B,d]
        h = F.relu(self.fc1(x_att))
        h = self.do(h)
        h = F.relu(self.fc2(h))
        h = self.do(h)
        logit = self.out(h)
        pd = torch.sigmoid(logit)         # [B,1]
        return pd, attn
