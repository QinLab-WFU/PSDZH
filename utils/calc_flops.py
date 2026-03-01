import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 你原代码里也有

import torch
from thop import profile, clever_format
from types import SimpleNamespace

# ✅ 这里 import 你这份网络代码所在的py文件
# 假设你这份代码文件名叫 zsl_vit.py，就写：
from network import build_model as build_my_model


def build_model_for_flops(device):
    # ✅ 构造 args：必须包含你 build_model(args) 用到的字段
    args = SimpleNamespace(
        backbone="evit_small",   # 对应 args.backbone.split("_") -> ("evit","small")
        n_classes=100,           # 这里随便填一个合理值即可（只影响分类头维度）
        dataset="awa2",           # "cub"/"awa2"/"sun" 三选一（决定 attr_dim）
        n_bits=32,               # 你要统计的 bit，比如 16/32/64/128
        device=device,
    )

    # ✅ 调你原来的 build_model(args, pretrained=True/False)
    # 建议 FLOPs 用 pretrained=False，避免下载权重失败影响统计
    net, _ = build_my_model(args, pretrained=False)
    return net


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model_for_flops(device).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224).to(device)

    # ✅ 关键：你模型 forward 有 labels/atts 参数，但它们默认是 None
    # 在 eval() + atts=None 时不会走语义注入分支，所以只喂 x 就行
    with torch.no_grad():
        flops, params = profile(model, inputs=(x,), verbose=False)

    flops, params = clever_format([flops, params], "%.3f")
    print(f"Params: {params}")
    print(f"FLOPs:  {flops}")


if __name__ == "__main__":
    main()
