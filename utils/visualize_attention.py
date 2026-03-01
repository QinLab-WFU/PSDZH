import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from network_Visualization import ZSLViT
import os
# -----------------------------
# Step 0: 初始化网络
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ZSLViT(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    dataset="cub",
    n_bits=312,
    keep_rate=[0.9]*12
)
net.eval()
net = net.to(device)

# -----------------------------
# Step 1: 读取图片
# -----------------------------
def load_image(path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1,3,H,W]

# -----------------------------
# Step 2: 叠加 attention map
# -----------------------------

# def overlay_attention(img_tensor, attn_list, token_idx_list=None, alpha_list=None, output_path=r"E:\ZSL\HASH-ZOO\PSDZH\images\antelope2_3,6.png"):
#     """
#     将 attention map 叠加到图像上并保存。
#
#     参数：
#         img_tensor: torch.Tensor, 形状 [1, C, H, W]，原图
#         attn_list: list of torch.Tensor, 每层的 cls attention
#         token_idx_list: list of torch.Tensor or None, 每层被选中 token 的索引
#         alpha_list: list of float or None, 每层 heatmap 的透明度
#         output_path: str, 保存路径
#     """
#     # ✅ 自动创建保存目录
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     # 转换原图到 [H, W, C] 并反归一化
#     img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
#     img = np.clip(img, 0, 1)
#
#     overlay = np.zeros_like(img[..., 0])
#
#     for i, cls_attn in enumerate(attn_list):
#         if cls_attn is None:
#             continue
#         n_patches = cls_attn.shape[-1]
#         h_w = int(np.sqrt(n_patches))
#         heatmap = cls_attn.cpu().numpy().reshape(h_w, h_w)
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
#
#         # 如果提供 token indices，则只保留被选中的 token
#         if token_idx_list is not None and token_idx_list[i] is not None:
#             mask = np.zeros_like(heatmap)
#             for idx in token_idx_list[i].cpu().numpy():
#                 y, x = divmod(idx, h_w)
#                 mask[y, x] = 1
#             heatmap = heatmap * mask
#
#         # resize heatmap到原图大小
#         heatmap_img = Image.fromarray(np.uint8(heatmap*255)).resize(
#             (img.shape[1], img.shape[0]), Image.BICUBIC
#         )
#         heatmap_img = np.array(heatmap_img)/255.0
#
#         # 透明度
#         a = alpha_list[i] if alpha_list is not None else 0.3
#         overlay = np.maximum(overlay, heatmap_img*a)
#
#     # 绘图并保存
#     plt.figure(figsize=(6,6))
#     plt.imshow(img)
#     plt.imshow(overlay, cmap='jet', alpha=0.6)
#     plt.axis('off')
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close()
#
#     print(f"Attention 可视化保存到 {output_path}")
# def overlay_attention(
#     img_tensor,
#     attn_list,
#     token_idx_list=None,
#     alpha_list=None,
#     topk_ratio=0.3,
#     selected_layers=None,
#     output_path=r"E:\ZSL\HASH-ZOO\PSDZH\images\wolf369.png"
# ):
#     import os
#     import numpy as np
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     import torch
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
#     img = np.clip(img, 0, 1)
#     overlay = np.zeros_like(img[..., 0])
#
#     if selected_layers is None:
#         selected_layers = list(range(len(attn_list)))
#
#     for i in selected_layers:
#         cls_attn = attn_list[i]
#         if cls_attn is None:
#             continue
#
#         # 如果多头维度存在，则取最大值
#         if cls_attn.dim() == 3:  # [B, H, N-1]
#             cls_attn = cls_attn.max(dim=1)[0]  # [B, N-1]
#
#         # batch=1 时，确保是一维
#         cls_attn = cls_attn.squeeze(0)  # [N-1]
#
#         # topk mask
#         if token_idx_list is not None and token_idx_list[i] is not None:
#             N = cls_attn.shape[0]
#             topk = max(1, int(N * topk_ratio))
#             topk_idx = torch.topk(cls_attn, topk, largest=True)[1]
#             mask = np.zeros(N, dtype=np.float32)
#             mask[topk_idx.cpu().numpy()] = 1.0
#             cls_attn = cls_attn.cpu().numpy() * mask
#         else:
#             cls_attn = cls_attn.cpu().numpy()
#
#         n_patches = cls_attn.shape[-1]
#         h_w = int(np.sqrt(n_patches))
#         heatmap = cls_attn.reshape(h_w, h_w)
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
#
#         heatmap_img = Image.fromarray(np.uint8(heatmap*255)).resize(
#             (img.shape[1], img.shape[0]), Image.BICUBIC
#         )
#         heatmap_img = np.array(heatmap_img)/255.0
#
#         a = alpha_list[i] if alpha_list is not None else 0.3
#         overlay = np.maximum(overlay, heatmap_img*a)
#
#     plt.figure(figsize=(6,6))
#     plt.imshow(img)
#     plt.imshow(overlay, cmap='jet', alpha=0.6)
#     plt.axis('off')
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close()
#     print(f"Attention 可视化保存到 {output_path}")
def overlay_attention(
    img_tensor,
    attn_list,
    token_idx_list=None,
    alpha_list=None,
    topk_ratio=0.3,
    selected_layers=None,
    output_path=r"E:\ZSL\HASH-ZOO\PSDZH\images\gorilla3.png"
):
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch
    from scipy.ndimage import gaussian_filter

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 原图反归一化
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    overlay = np.zeros_like(img[..., 0])

    if selected_layers is None:
        selected_layers = list(range(len(attn_list)))

    for i in selected_layers:
        cls_attn = attn_list[i]
        if cls_attn is None:
            continue

        # 多头注意力取最大
        if cls_attn.dim() == 3:  # [B, H, N-1]
            cls_attn = cls_attn.max(dim=1)[0]  # [B, N-1]

        cls_attn = cls_attn.squeeze(0)  # batch=1

        # top-k mask
        if token_idx_list is not None and token_idx_list[i] is not None:
            N = cls_attn.shape[0]
            topk = max(1, int(N * topk_ratio))
            topk_idx = torch.topk(cls_attn, topk, largest=True)[1]
            mask = np.zeros(N, dtype=np.float32)
            mask[topk_idx.cpu().numpy()] = 1.0
            cls_attn = cls_attn.cpu().numpy() * mask
        else:
            cls_attn = cls_attn.cpu().numpy()

        n_patches = cls_attn.shape[-1]
        h_w = int(np.sqrt(n_patches))
        heatmap = cls_attn.reshape(h_w, h_w)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

        # 高斯加权中心化，让中间区域更亮
        y, x = np.meshgrid(np.linspace(-1,1,h_w), np.linspace(-1,1,h_w))
        d = np.sqrt(x*x + y*y)
        sigma = 0.5
        gauss = np.exp(-(d**2)/(2*sigma**2))
        heatmap = heatmap * gauss

        # 高斯模糊平滑
        heatmap = gaussian_filter(heatmap, sigma=1)

        # resize heatmap到原图大小
        heatmap_img = Image.fromarray(np.uint8(heatmap*255)).resize(
            (img.shape[1], img.shape[0]), Image.BICUBIC
        )
        heatmap_img = np.array(heatmap_img)/255.0

        a = alpha_list[i] if alpha_list is not None else 0.3
        overlay = np.maximum(overlay, heatmap_img*a)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(overlay, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Attention 可视化保存到 {output_path}")

# -----------------------------
# Step 3: 渐进式 attention 可视化
# -----------------------------
def visualize_progressive_combined(net, img_tensor, keep_rate=0.9, label_tensor=None, atts_tensor=None,
                                   layers_to_show=[3,6,9], device='cuda'):
    net.eval()
    img_tensor = img_tensor.to(device)
    if label_tensor is not None:
        label_tensor = label_tensor.to(device)
    if atts_tensor is not None:
        atts_tensor = atts_tensor.to(device)

    # 保证 keep_rate 是列表
    if not isinstance(keep_rate, (tuple, list)):
        keep_rate = [keep_rate] * net.depth

    with torch.no_grad():
         _, _, _, _, attn_maps, token_indices = net(img_tensor, keep_rate=keep_rate,
                                                     labels=label_tensor, atts=atts_tensor)

    cls_attn_list = []
    token_idx_list = []
    for l in layers_to_show:
        cls_attn_list.append(attn_maps[l] if l in attn_maps else None)
        token_idx_list.append(token_indices[l] if l in token_indices else None)

    overlay_attention(img_tensor, cls_attn_list, token_idx_list)

# -----------------------------
# Step 4: 示例调用
# -----------------------------
if __name__ == "__main__":
    img_path = r"E:\ZSL\HASH-ZOO\PSDZH\images\gorilla.jpg"  # 替换为你的图片路径
    img_tensor = load_image(img_path).to(device)

    # 假设标签为0
    label_tensor = torch.tensor([0]).to(device)
    # 属性向量，可随机生成示例
    attr_dim = net.S2V[0].fc[0].in_features
    atts_tensor = torch.randn(1, attr_dim).to(device)

    visualize_progressive_combined(net, img_tensor, keep_rate=[0.9]*12,
                                   label_tensor=label_tensor, atts_tensor=atts_tensor,
                                   layers_to_show=[3], device=device)
