# ddpm_train_fixed_amp.py
"""
DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）的完整训练示例。

核心思想（小白版）：
    扩散模型是目前最主流的图像生成方法（Stable Diffusion、DALL·E 等的核心）。它分两个方向相反的过程：

    1. 前向加噪（forward / q_sample）：把一张真实图片逐步加噪声，分成 T=1000 步，
       每加一点，直到第 T 步变成完全的高斯噪声（纯雪花）。这个过程不需学习，是固定的数学公式。
    2. 反向去噪（reverse / p_sample）：训练一个神经网络（这里是 UNet），学会「把噪声图去掉一点噪声」。
       生成时从一张纯噪声出发，反复调用它去噪 T 次，就能“雕刻”出一张新图片。

    训练目标（关键）：不直接预测干净图片，而是「预测加进去的噪声 eps」。
        给一张图随机选一个时间步 t，用 q_sample 加上噪声得到 x_t，
        让 UNet 看着 x_t 和 t 去猜“加了什么噪声”，用 MSE 对比真实噪声。

    关键零件：
        - beta schedule：定义每一步加多少噪声（从很小逐渐变大）。
        - alpha_cumprod：累乘系数，用于一步直接从原图算出任意时刻 t 的加噪结果。
        - 时间嵌入 SinusoidalPosEmb：把时间步 t（一个整数）编码成向量，告诉 UNet“现在是第几步”，
          因为不同噪声程度需要不同的去噪力度。
        - UNet：带残差块与自注意力的去噪网络，下采样提取特征再上采样恢复尺寸，
          中间用 skip connection（跳跃连接）保留细节。

    AMP（自动混合精度）：在有 GPU 时用半精度计算，省显存、提速。

流程：真实图片 --(随机 t + q_sample 加噪)--> 噪声图 x_t --UNet预测噪声--> 与真实噪声算 MSE
生成：纯噪声 --(p_sample 逐步去噪 T 次)--> 新图片
"""
import os
import math
from datetime import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# ---------- User config (kept from your original) ----------
# 以下为训练的全局配置（超参数）
data_root = "/mnt/d/data/face/img/img_align_celeba"  # CelebA 人脸数据集目录（需自行准备）
save_dir = "./ddpm_checkpoints"                       # 保存权重和生成样本的目录
os.makedirs(save_dir, exist_ok=True)
batch_size = 8            # 每批图片数（UNet 较大、显存占用高，故 batch 小）
lr = 1e-5                 # 学习率（扩散模型通常用很小的学习率）
num_epochs = 100          # 总训练轮数
image_size = 104          # 输入图片尺寸 104x104
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4           # 数据加载子进程数
pin_memory = True         # 锁页内存，加快 CPU->GPU 传输
sample_every = 1          # 每隔多少轮生成一次样本并保存权重
num_sample_images = 8     # 每次生成多少张样本图
base_ch = 128             # UNet 基础通道数
T = 1000  # diffusion timesteps（扩散总步数，即加噪/去噪分多少步）

# Use AMP if CUDA is available
# 仅在有 GPU 且支持 amp 时启用混合精度（CPU 上无法用）
use_amp = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

# ---------- Data ----------
# 图像预处理：缩放 + 中心裁剪到 image_size，ToTensor 归一化到 [0,1]
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # [0,1]
])


class CelebADataset(Dataset):
    """CelebA 人脸数据集：把目录下所有 .jpg 图片包装成 Dataset。"""
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))  # 所有图片路径
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # scale to [-1,1]
        # 把像素从 [0,1] 线性变换到 [-1,1]：扩散模型在均值 0 附近工作更稳定（噪声是标准正态）
        img = img * 2.0 - 1.0
        return img


dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)


# ---------- Diffusion schedule utilities ----------
# 下面这一段预先算好扩散过程需要的各种系数，是整个 DDPM 的数学核心。
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """线性 beta 调度：返回长度为 timesteps 的 beta 序列，从 beta_start 线性增加到 beta_end。

    beta_t 控制第 t 步加多少噪声：越往后 beta 越大（加得越狠），直到图片完全变成噪声。
    """
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(T).to(device)  # shape [T]，每一步的 beta
alphas = 1.0 - betas                        # alpha_t = 1 - beta_t
alpha_cumprod = torch.cumprod(alphas, dim=0)  # \bar{\alpha}_t，alpha 的累乘，用于一步直接算任意时刻的加噪结果
# alpha_cumprod_prev：把累乘序列向后错一位（开头补 1.0），即 \bar{\alpha}_{t-1}，采样时要用
alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]], dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)                        # sqrt(\bar{\alpha}_t)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)        # sqrt(1-\bar{\alpha}_t)

# precompute terms for sampling
# 后验方差：反向采样每一步需要加的随机噪声的方差（除 t=0 外都保留一点随机性）
posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)


# ---------- Time embedding ----------
class SinusoidalPosEmb(nn.Module):
    """正弦位置编码（与 Transformer 里的一样）：把时间步 t（一个整数）编码成一个连续向量。

    为什么要编码？因为 UNet 需要知道“当前是第几步/噪声多严重”，才能决定去噪力度。
    直接把整数 t 丢进去不好，用不同频率的 sin/cos 组合成一个富信息、可区分的向量。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 输出嵌入向量的维度

    def forward(self, t):
        # t: (B,) longs，一批图片各自的时间步
        device = t.device
        half = self.dim // 2
        # 构造一组从大到小的频率（等比数列），低维用高频、高维用低频
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]  # t 与各频率相外积，(B, half)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # sin/cos 拼接成 (B, dim)
        if self.dim % 2 == 1:  # 维度为奇数时补一列 0
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


# ---------- Enhanced UNet with Residual Blocks + Attention ----------
class ResidualBlock(nn.Module):
    """残差块：UNet 的基本单元。两层卷积 + 把时间嵌入注入进来 + 残差连接。

    为什么要把时间嵌入加进来？因为去噪行为依赖于当前噪声程度（时间步 t），
    把 t_emb 加到特征图上，相当于告诉卷积“现在噪声多严重”。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        # 第一层：GroupNorm 归一化 -> SiLU 激活 -> 3x3 卷积（把通道从 in_ch 变 out_ch）
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        # 把时间嵌入向量映射到 out_ch 维，以便加到特征图上
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        # 第二层：归一化 -> 激活 -> Dropout -> 3x3 卷积
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        # 残差连接：若输入输出通道不同，用 1x1 卷积对齐通道；否则直接恒等映射
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)      # 保存残差分支（对齐通道）
        h = self.conv1(x)                     # 第一层卷积
        t_emb = self.time_emb_proj(t_emb)     # 时间嵌入投影到 out_ch 维
        h = h + t_emb[:, :, None, None]       # 把时间信息加到特征图上（None,None 把 (B,C) 扩到 (B,C,1,1) 以便广播）
        h = self.conv2(h)                     # 第二层卷积
        return h + residual                   # 加上残差，缓解梯度消失


# ---------- Self-Attention Layer ----------
class SelfAttention2D(nn.Module):
    """二维自注意力层：让图片上每个像素位置都能“看到”其他位置，捕获全局关系。

    卷积只看局部邻域，注意力能补充全局信息（尤其在分辨率小的深层）。采用多头注意力 + 残差连接。
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads  # 多头注意力的头数
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)  # 1x1 卷积一次性算出 q,k,v（故通道*3）
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 输出投影

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)  # 沿通道维切成 q,k,v 三份
        # reshape to (B, heads, C//heads, H*W)
        # 把每个头分开，并把空间 H*W 展平成序列长度（每个像素当一个 token）
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # 注意力分数 = softmax(q^T k / sqrt(d))，衡量每个位置对其他位置的关注度
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        # 用注意力权重对 v 加权求和，得到每个位置的新表示，再 reshape 回 (B,C,H,W)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out  # residual connection（残差连接）


class DownBlock(nn.Module):
    """下采样块：若干个残差块 + （可选）注意力 + （可选）下采样（尺寸减半）。

    返回 (x, skips)：skips 保存每个残差块的输出，用于上采样时的跳跃连接（保留细节）。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        # 堆叠 num_blocks 个残差块（第一个负责把通道从 in_ch 变 out_ch，其余保持 out_ch）
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()  # 不用注意力时用恒等映射占位
        # 下采样：步长 2 的卷积把长宽减半（不用时 Identity）
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)   # 保存中间输出供跳跃连接
        x = self.attn(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    """上采样块：（可选）上采样（尺寸翻倍） + 拼接跳跃特征 + 若干残差块 + （可选）注意力。"""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        # 上采样：转置卷积把长宽放大一倍（不用时 Identity）
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        # 注意输入通道是 in_ch + out_ch：因为要和下采样时的 skip 特征拼接（cat）后再卷积
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)
        for block in self.blocks:
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)  # 与对应的下采样特征沿通道拼接（UNet 的核心）
            x = block(x, t_emb)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """中间块：位于 UNet 最底层（分辨率最小），若干残差块 + 注意力，不改变尺寸。"""
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)  # 加注意力层

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x


# ---------- Full Enhanced UNet with Attention ----------
class EnhancedUNet(nn.Module):
    """完整的去噪 UNet：输入噪声图 x_t 和时间步 t，输出预测的噪声。

    结构是经典的 U 形：
        下采样（down1~down4，尺寸逐级减半、通道逐级变多） -> 中间块 mid
        -> 上采样（up4~up1，尺寸逐级恢复），上采样时与下采样的 skip 特征拼接。
    时间步 t 先经 time_mlp 编码成时间嵌入 t_emb，注入到每个残差块里。
    """
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        # 时间嵌入网络：正弦编码 -> 两层 MLP，把时间步 t 变成一个 time_emb_dim 维向量
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)  # 入口卷积：3通道 -> base_ch

        # Down blocks (添加注意力在 deeper layers)
        # 下采样：通道 base -> 2base -> 4base -> 8base；down1 不降尺寸，最深的 down4 加注意力
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        # Middle block
        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # Up blocks (同样添加注意力)
        # 上采样：通道 8base -> 4base -> 2base -> base；up4 加注意力，up1 不再上采样（恢复原尺寸）
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        # 输出层：归一化 -> 激活 -> 3x3 卷积把通道变回 in_ch（即预测的噪声图，与输入同尺寸）
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)   # 先把时间步编码成向量
        x = self.init_conv(x)      # 入口卷积

        # 下采样阶段：把所有残差块的输出收集到 skips 里（供上采样拼接）
        skips = []
        x, s1 = self.down1(x, t_emb); skips.extend(s1)
        x, s2 = self.down2(x, t_emb); skips.extend(s2)
        x, s3 = self.down3(x, t_emb); skips.extend(s3)
        x, s4 = self.down4(x, t_emb); skips.extend(s4)

        x = self.mid(x, t_emb)     # 中间块

        # 上采样阶段：每步从 skips 里 pop 出对应特征拼接（顺序与下采样相反）
        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)       # 输出预测的噪声


# ---------- Diffusion forward q_sample ----------
def q_sample(x_start, t, noise=None):
    """前向加噪：一步直接算出原图 x_start 在时刻 t 的加噪结果 x_t（不需逐步迭代）。

    公式：x_t = sqrt(\bar{\alpha}_t) * x_start + sqrt(1-\bar{\alpha}_t) * noise
        前半部分是“保留的原图信号”，后半部分是“加入的噪声”，t 越大噪声占比越高。
    x_start: (B,C,H,W) in [-1,1]
    t: tensor of shape (B,) with values in [0,T-1]，每张图可以取不同的时间步
    返回：(加噪后的 x_t, 真实噪声 noise)，noise 作为训练目标。
    """
    if noise is None:
        noise = torch.randn_like(x_start)  # 未提供则随机采标准正态噪声
    # 用 t 作下标取出每张图对应的系数，view(-1,1,1,1) 是为了能广播到 (B,C,H,W)
    sqrt_alpha_cumprod_t = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise, noise


# ---------- Loss (predict noise) ----------
def p_losses(model, x_start, t):
    """训练损失：让 UNet 预测的噪声尽量接近真实加入的噪声。

    步骤：给原图加噪 -> UNet 预测噪声 -> 与真实噪声算 MSE。
    """
    x_noisy, noise = q_sample(x_start, t)      # 加噪，得到噪声图和真实噪声
    predicted_noise = model(x_noisy, t)        # UNet 预测“加了什么噪声”
    loss = F.mse_loss(predicted_noise, noise, reduction='mean')  # 预测噪声与真实噪声的均方误差
    return loss


# ---------- Sampling (ancestral sampling from DDPM) ----------
@torch.no_grad()
def p_sample(model, x_t, t):
    """反向去噪一步：从时刻 t 的 x_t 推出 x_{t-1}（噪声少一点的图）。

    这是 DDPM 的“祖先采样”：先用 UNet 预测噪声，算出去噪后的均值 mu，
    再加上一点随机噪声（t=0 时不加），得到上一步的结果。
    """
    # 当前步的参数
    alpha_t = alphas[t]
    alpha_cumprod_t = alpha_cumprod[t]
    alpha_cumprod_prev_t = alpha_cumprod_prev[t]

    # 预测噪声 ε（整个 batch 用同一个 t，故用 full 造一个全为 t 的向量）
    pred_noise = model(x_t, torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device))

    # ---- 计算均值 μ ----
    # μ = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-\bar{\alpha}_t) * 预测噪声)，即从 x_t 中减掉预测的噪声
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

    mu = (1.0 / sqrt_alpha_t) * (
        x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
    )

    # ---- 计算方差 σ² ----
    # 后验方差，控制这一步要重新加多少随机噪声（保留生成的多样性）
    sigma2 = ((1.0 - alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)
    sigma = torch.sqrt(sigma2)

    # ---- 采样 x_{t-1} ----
    if t == 0:
        return mu  # 最后一步（t=0）直接返回均值，不再加噪声，得到干净图片
    else:
        noise = torch.randn_like(x_t)
        return mu + sigma.view(1, 1, 1, 1) * noise  # 均值 + 随机噪声



@torch.no_grad()
def sample_loop(model, batch_size, device):
    """完整生成（采样）循环：从纯噪声出发，从 t=T-1 到 0 逐步去噪，最终得到新图片。"""
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)  # 起点：纯高斯噪声
    for t in reversed(range(T)):  # 从最后一步往前逐步去噪
        # use autocast during sampling for better performance on GPU
        if use_amp:
            with torch.cuda.amp.autocast():
                x = p_sample(model, x, t)
        else:
            x = p_sample(model, x, t)
    # x in [-1,1], convert to [0,1]
    # 模型输出在 [-1,1]，转回 [0,1] 才能当图片保存，并 clamp 防止越界
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x


# ---------- Utilities ----------
def save_checkpoint(model, optim, epoch, path, scaler=None):
    """保存训练现场（权重 + 优化器 + 轮数 + 可选的 AMP scaler 状态）。"""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    if scaler is not None:  # 用了 AMP 就把 GradScaler 状态也存下来（恢复训练时保证缩放一致）
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, path)


def save_image_grid(tensor, filename, nrow=8):
    """把一批图片张量拼成网格保存到 filename。"""
    # tensor expected in [0,1]
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


# ---------- Training ----------
def train():
    """DDPM 训练主循环：含断点续训、AMP 混合精度、梯度裁剪、定期采样保存。"""
    model = EnhancedUNet(in_ch=3, base_ch=base_ch, time_emb_dim=512, num_res_blocks=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # AMP scaler
    # GradScaler：混合精度时把损失放大再反向，避免半精度下梯度过小而溢出（enabled 由 use_amp 控制）
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1

    # optionally load last checkpoint if present (and scaler state)
    # 断点续训：若存在上次保存的 ddpm_last.pth，则加载模型/优化器/scaler 状态，从上次轮数继续
    last_ckpt = os.path.join(save_dir, "ddpm_last.pth")
    if os.path.exists(last_ckpt):
        print("Loading checkpoint:", last_ckpt)
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        if "scaler_state" in ck and use_amp:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                print("Warning: failed to load scaler state (version mismatch?)")
        start_epoch = ck["epoch"] + 1
        print("Resumed from epoch", start_epoch)

    global_step = 0
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device)  # in [-1,1]
            bs = imgs.size(0)

            # sample t for each image in the batch (uniform 0..T-1)
            # 为 batch 里每张图随机抽一个时间步 t（均匀分布 0~T-1），让模型学会处理各种噪声程度
            t = torch.randint(0, T, (bs,), device=device).long()

            optim.zero_grad()

            # forward + loss within autocast if AMP enabled
            if use_amp:
                with torch.cuda.amp.autocast():   # 自动混合精度：内部用半精度加速前向
                    loss = p_losses(model, imgs, t)
                # scale -> backward
                scaler.scale(loss).backward()     # 先缩放损失再反向，防止梯度下溢
                # unscale before clipping
                scaler.unscale_(optim)            # 裁剪前把梯度缩放回来（才能按真实大小裁剪）
                # Gradient clipping for stability (unscaled grads)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                scaler.step(optim)                # 等价于 optim.step()，但会处理 inf/NaN
                scaler.update()                   # 更新缩放因子
            else:
                # 无 GPU 时的普通训练分支
                loss = p_losses(model, imgs, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # 每 50 个 batch 打印一次损失
            if batch_idx % 50 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch}/{num_epochs} Batch {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.6f}")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"=== Epoch {epoch} finished. Avg loss: {avg_loss:.6f} ===")

        # sample images every few epochs
        if epoch % sample_every == 0 or epoch == 1:
            # save checkpoint (include scaler state if using AMP)
            # 保存本轮权重（另存一份 ddpm_last.pth 供断点续训）
            ckpt_path = os.path.join(save_dir, f"ddpm_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path, scaler if use_amp else None)
            # update last
            save_checkpoint(model, optim, epoch, last_ckpt, scaler if use_amp else None)

            # 切到评估模式，用当前模型生成一批图片，直观看生成效果（从噪声逐渐变清晰）
            model.eval()
            with torch.no_grad():
                samples = sample_loop(model, num_sample_images, device)
            # save grid
            save_image_grid(samples, os.path.join(save_dir, f"sample_epoch{epoch}.png"), nrow=8)
            model.train()

    print("Training complete.")


def merge_models(model_checkpoints, output_path, merge_method='average'):
    """
    合并多个模型检查点

    Args:
        model_checkpoints: 模型检查点路径列表
        output_path: 合并后模型的保存路径
        merge_method: 合并方法，'average'为平均权重，'ema'为指数移动平均
    """
    print(f"开始合并 {len(model_checkpoints)} 个模型检查点...")

    # 加载第一个模型作为基础
    base_checkpoint = torch.load(model_checkpoints[0], map_location='cpu')
    merged_state_dict = base_checkpoint['model_state'].copy()

    if merge_method == 'average':
        # 平均权重
        for checkpoint_path in model_checkpoints[1:]:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            for key in merged_state_dict.keys():
                merged_state_dict[key] += checkpoint['model_state'][key]

        for key in merged_state_dict.keys():
            merged_state_dict[key] = merged_state_dict[key] / len(model_checkpoints)

    elif merge_method == 'ema':
        # 指数移动平均 (EMA)，越新的模型权重越大
        alpha = 0.9  # EMA衰减因子
        weight = 1.0

        for i, checkpoint_path in enumerate(model_checkpoints[1:], 1):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            current_weight = weight * (alpha ** (len(model_checkpoints) - i - 1))

            for key in merged_state_dict.keys():
                merged_state_dict[key] = (merged_state_dict[key] * (1 - current_weight) +
                                          checkpoint['model_state'][key] * current_weight)

    # 保存合并后的模型
    merged_checkpoint = {
        'model_state': merged_state_dict,
        'epoch': f"merged_from_{len(model_checkpoints)}_models",
        'merge_method': merge_method
    }

    torch.save(merged_checkpoint, output_path)
    print(f"合并完成！模型已保存至: {output_path}")

    return merged_state_dict


def generate_images_with_merged_model(model_path, num_images=16, output_dir="./merged_model_samples"):
    """使用合并后的模型生成图片"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载合并模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型并加载权重
    model = EnhancedUNet(in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print("开始生成图片...")
    with torch.no_grad():
        samples = sample_loop(model, num_images, device)

    # 保存生成的图片
    output_filename = os.path.join(output_dir, f"merged_model_samples.png")
    save_image_grid(samples, output_filename, nrow=4)
    print(f"图片已保存至: {output_filename}")

    # 同时保存一些中间结果（可选）
    print("生成多组图片...")
    for i in range(3):
        with torch.no_grad():
            samples = sample_loop(model, num_images, device)
        output_filename = os.path.join(output_dir, f"merged_model_samples_set_{i + 1}.png")
        save_image_grid(samples, output_filename, nrow=4)
        print(f"第 {i + 1} 组图片已保存")


if __name__ == "__main__":
    print("Starting DDPM training on device:", device)
    print("AMP enabled:", use_amp)
    print("Dataset size:", len(dataset))
    train()
