"""CLIP ViT-L/14@336 시각부 — 순수 PyTorch 구현(외부 패키지 버전 차이에 영향받지 않도록 직접 작성).
학습 때 특징을 뽑은 open_clip VisionTransformer(활성화 GELU, 클래스 토큰 풀링, ln_post 후 proj)와 같은 계산.
state_dict 키 이름도 같게 맞춰 model/b2_visual_fp16.safetensors 를 그대로 읽는다.
"""
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(d, 4 * d))
        self.mlp.add_module("gelu", nn.GELU())
        self.mlp.add_module("c_proj", nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class _Tr(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.resblocks = nn.ModuleList([_Block(d, heads) for _ in range(layers)])

    def forward(self, x):
        for b in self.resblocks:
            x = b(x)
        return x


class ViTL14_336(nn.Module):
    def __init__(self, d=1024, layers=24, heads=16, patch=14, image=336, out=768):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=patch, stride=patch, bias=False)
        s = d ** -0.5
        self.class_embedding = nn.Parameter(s * torch.randn(d))
        self.positional_embedding = nn.Parameter(s * torch.randn((image // patch) ** 2 + 1, d))
        self.ln_pre = nn.LayerNorm(d)
        self.transformer = _Tr(d, layers, heads)
        self.ln_post = nn.LayerNorm(d)
        self.proj = nn.Parameter(s * torch.randn(d, out))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        return x[:, 0] @ self.proj
