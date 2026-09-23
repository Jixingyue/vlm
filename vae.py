"""
VAE（Variational AutoEncoder，变分自编码器）的最小实现示例。

核心思想（小白版）：
    VAE 是一种「生成模型」，目标是学会数据（这里是人脸图片）的分布，从而能「凭空生成」新图片。
    它由两部分组成：
      1. 编码器 Encoder：把一张图片压缩成一个「潜在向量（latent）」，但不是压成一个固定的点，
         而是压成一个「概率分布」（用均值 mu 和方差 logvar 描述）。这样潜在空间更平滑、连续。
      2. 解码器 Decoder：从潜在向量再还原（重建）出图片。

    与普通自编码器的关键区别 —— 重参数化技巧（reparameterize）：
        直接「从分布里采样」这个操作不可导，没法反向传播。于是把采样改写成：
            z = mu + eps * std   （eps 是标准正态噪声）
        这样随机性被挪到了 eps 上，mu/std 仍然可导，训练就能正常进行了。

    训练目标（损失函数）由两项组成：
        - 重建损失 recon：让还原出来的图片尽量接近原图（这里用 MSE 均方误差）。
        - KL 散度 kld：约束编码器输出的分布尽量接近标准正态分布 N(0,1)，
          保证潜在空间规整、可采样生成。
        总损失 = recon + kld。

    生成新图片：直接从标准正态分布随机采一个 z，喂给解码器 decode(z) 即可。

流程：图片 -> 编码器 -> (mu, logvar) -> 重参数化采样 z -> 解码器 -> 重建图片 -> 计算 recon+KL 损失
"""

import os
from datetime import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# ========== Paths ==========
# data_root：CelebA 人脸数据集所在目录（需自行准备，里面是 .jpg 图片）
# save_dir：训练过程中保存模型权重(checkpoint)和生成样本图片的目录
data_root = "/mnt/d/data/face/img/img_align_celeba"
save_dir = "./vae_test_checkpoints"
os.makedirs(save_dir, exist_ok=True)  # 目录不存在则自动创建

# ========== Hyperparams ==========
batch_size = 1024        # 每批训练多少张图（VAE 较轻量，可用大 batch）
lr = 2e-4                # 学习率
num_epochs = 300         # 总共训练多少轮
latent_dim = 128         # 潜在向量 z 的维度（把图片压缩成 128 个数）
sample_every = 5         # 每隔多少轮保存一次样本图片和权重
num_sample_images = 8    # 每次可视化生成/重建多少张图
image_size = 64          # 输入图片会被缩放裁剪到 64x64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU，否则用 CPU

# 图像预处理：先缩放到短边为 image_size，再从中心裁剪成 image_size x image_size 正方形，
# 最后 ToTensor 把像素值归一化到 [0,1] 区间（VAE 解码器输出用了 Sigmoid，正好对应 [0,1]）
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])


# ========== Dataset ==========
class CelebADataset(Dataset):
    """CelebA 人脸数据集封装：把文件夹里的所有 .jpg 图片包装成 PyTorch 可用的 Dataset。"""
    def __init__(self, root, transform=None):
        self.root = root
        # glob 找出目录下所有 .jpg 文件并排序，得到图片路径列表
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)  # 数据集大小 = 图片数量

    def __getitem__(self, idx):
        # DataLoader 每次取一个下标 idx，这里返回对应的那张（预处理后的）图片
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")  # 打开图片并统一转成 3 通道 RGB
        if self.transform:
            img = self.transform(img)
        return img


dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=8, pin_memory=True)


# ========== Model ==========
class ConvVAE(nn.Module):
    """卷积 VAE 模型：编码器（下采样卷积） + 潜在分布 + 解码器（上采样反卷积）。

    参数：latent_dim=潜在向量维度，ch=基础通道数，image_size=输入图片边长。
    图片尺寸变化（以 image_size=64, ch=64 为例）：
        输入 (B,3,64,64) --编码器4次下采样--> (B,512,4,4) --展平--> (B,8192)
        --fc_mu/fc_logvar--> mu,logvar (B,128) --采样 z--> (B,128)
        --fc_dec--> (B,8192) --reshape--> (B,512,4,4) --解码器4次上采样--> (B,3,64,64)
    """
    def __init__(self, latent_dim=128, ch=64, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.image_size = image_size

        # Encoder（编码器）：4 个步长为 2 的卷积，每次把图片长宽缩小一半（下采样），通道数递增
        # Conv2d(3, ch, 4, 2, 1) 中 4=卷积核, 2=步长stride, 1=填充padding，这样尺寸正好减半
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),   # BatchNorm 归一化，稳定训练、加快收敛
            nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True)
        )

        # 用一个全 0 的假图片跑一遍编码器，自动算出展平后的特征维度 feat_dim，
        # 这样后面全连接层的输入维度就不用手写（不同 image_size 都能自适应）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_dim = self.enc(dummy).view(1, -1).size(1)

        # 编码器输出两个向量：均值 mu 和 对数方差 logvar（用 logvar 而非方差是为了数值稳定、保证方差>0）
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        # 解码器入口：把潜在向量 z 映射回编码器最后的特征维度
        self.fc_dec = nn.Linear(latent_dim, feat_dim)
        self._feat_shape = self.enc(dummy).shape[1:]  # 记录解码前的特征图形状 (C,H,W)，用于 reshape

        # Decoder（解码器）：4 个转置卷积（反卷积），每次把长宽放大一倍（上采样），与编码器对称
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
            nn.Sigmoid()  # 添加Sigmoid激活，把输出像素值压缩到 [0,1]，对应图片归一化区间
        )

    def encode(self, x):
        """编码：图片 x -> 潜在分布的参数 (mu, logvar)。"""
        h = self.enc(x)                 # 卷积下采样得到特征图
        h = h.view(h.size(0), -1)       # 展平成 (B, feat_dim)
        mu = self.fc_mu(h)              # 潜在分布的均值
        logvar = self.fc_logvar(h)      # 潜在分布的对数方差
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从 N(mu, std^2) 中采样一个 z，但保证可导。

        做法：z = mu + eps * std，其中 eps ~ N(0,1) 是外部噪声。
        这样随机性只在 eps 上，mu/std 对参数的梯度可以正常回传。
        """
        std = torch.exp(0.5 * logvar)   # 由对数方差还原标准差：std = exp(0.5*log(var)) = sqrt(var)
        eps = torch.randn_like(std)     # 采样标准正态噪声
        return mu + eps * std

    def decode(self, z):
        """解码：潜在向量 z -> 重建图片。生成新图片时也会直接调用它。"""
        h = self.fc_dec(z)                       # z 映射回特征维度 (B, feat_dim)
        h = h.view(h.size(0), *self._feat_shape) # reshape 回特征图形状 (B,C,H,W)
        x_recon = self.dec(h)                    # 反卷积上采样还原成图片
        return x_recon

    def forward(self, x):
        """完整前向：编码 -> 采样 -> 解码，返回重建图、mu、logvar（后两者用于算 KL 损失）。"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ========== Loss ==========
def vae_loss(recon_x, x, mu, logvar):
    """VAE 总损失 = 重建损失 + KL 散度。

    - 重建损失 recon_loss：重建图 recon_x 与原图 x 的 MSE（用 sum 求和，而非均值）。
    - KL 散度 kld：衡量编码器输出的分布 N(mu, var) 与标准正态 N(0,1) 的差异，
      公式 -0.5 * sum(1 + logvar - mu^2 - exp(logvar))，它促使潜在空间规整。
    返回：(总损失, 重建损失, KL损失)，后两项仅用于日志打印。
    """
    # 使用 MSE 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld


# ========== Utilities ==========
def save_checkpoint(model, optim, epoch, path):
    """保存训练现场（权重 + 优化器状态 + 轮数）到 path，方便以后恢复训练。"""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),     # 模型参数
        "optim_state": optim.state_dict()      # 优化器状态（如动量）
    }
    torch.save(state, path)


def save_image_grid(tensor, filename, nrow=8):
    """把一批图片张量拼成网格图保存到 filename（方便直观查看）。"""
    tensor = torch.clamp(tensor, 0, 1)  # 像素值限在 [0,1]，避免保存图片时出错
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


# ========== Training ==========
def train():
    """VAE 训练主循环：逐轮、逐 batch 前向算损失 -> 反向传播 -> 更新参数，并定期保存样本。"""
    model = ConvVAE(latent_dim=latent_dim,image_size=image_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()  # 切换到训练模式（启用 Dropout/BatchNorm 的训练行为）
        # 累计本轮的总损失、重建损失、KL 损失，用于轮末打印平均值
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)  # 数据搬到 GPU/CPU
            optim.zero_grad()                          # 清空上一批的梯度
            recon_imgs, mu, logvar = model(imgs)       # 前向：得到重建图与潜在分布参数
            loss, recon_l, kld = vae_loss(recon_imgs, imgs, mu, logvar)  # 算总损失
            loss.backward()                            # 反向传播算梯度
            optim.step()                               # 更新参数

            # 累加各项损失（用于统计）
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kld += kld.item()
            global_step += 1

            # 每 100 个 batch 打印一次当前损失，方便观察训练进度
            if batch_idx % 100 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch}/{num_epochs} Batch {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.4f} "
                      f"(recon {recon_l.item():.4f}, kld {kld.item():.4f})")

        # 本轮结束，打印平均损失（除以样本总数，因为损失用的是 sum）
        n_samples = len(loader.dataset)
        print(f"=== Epoch {epoch} finished. Avg loss: {epoch_loss / n_samples:.4f} "
              f"(recon {epoch_recon / n_samples:.4f}, kld {epoch_kld / n_samples:.4f}) ===")

        # 保存样本
        if epoch % sample_every == 0 or epoch == 1:
            # 保存检查点
            ckpt_path = os.path.join(save_dir, f"vae_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path)
            model.eval()  # 切换到评估模式（固定 BatchNorm/Dropout）再采样
            with torch.no_grad():  # 采样不需梯度，关闭以省显存
                # 重建样本：取一批图过一遍模型，把原图与重建图拼在一起保存对比
                imgs = next(iter(loader))
                imgs = imgs.to(device)[:num_sample_images]
                recon_imgs, _, _ = model(imgs)
                combined = torch.cat([imgs, recon_imgs], dim=0)  # 不再需要clamp
                save_image_grid(combined, os.path.join(save_dir, f"recon_epoch{epoch}.png"), nrow=8)

                # 生成样本：直接从标准正态分布随机采 z，用解码器生成全新图片
                z = torch.randn(num_sample_images, latent_dim).to(device)
                samples = model.decode(z)
                save_image_grid(samples, os.path.join(save_dir, f"sample_epoch{epoch}.png"), nrow=8)
            model.train()  # 采样完毕切回训练模式

    print("Training complete.")


if __name__ == "__main__":
    print("Starting training on device:", device)
    print("Dataset size:", len(dataset))
    train()
