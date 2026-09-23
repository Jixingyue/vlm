"""
MAE（Masked AutoEncoder，掩码自编码器）的最小实现示例。

核心思想（小白版）：
    这也是一种自监督学习。把一张图片切成很多小块（patch），然后随机“遮住”大部分小块（默认 75%），
    让编码器只看得到剩下的小块，再用解码器把被遮住的小块“重建”出来。
    训练目标就是：重建出来的小块越接近原图被遮住的小块越好（用 MSE 均方误差衡量）。
    通过“猜被遮住的部分”，模型被迫学会图像真正的语义信息。

流程：图片 -> 切 patch -> 随机遮住一部分 -> 编码器 -> 解码器 -> 重建 -> 与被遮原图算损失
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


def mse_loss(reconstructed_patches, patches, mask_indices):
    """只在“被遮住的 patch”上计算重建损失（均方误差 MSE）。

        Parameters:
        - reconstructed_patches: Tensor of shape (B, 196, 768), reconstructed patch embeddings.（解码器重建出的 patch）
        - patches: Tensor of shape (B, 196, 768), original patch embeddings.（原图的 patch，作为重建目标）
        - mask_indices: LongTensor of shape (196,), indices of the masked patches.（被遮住那些 patch 的下标）
    """
    B = reconstructed_patches.size(0)

    # Only consider the masked patches, gather函数用来从patches中选择那些被mask的patches出来
    # 虽然有多个batch，但每个batch上都是那些位置的元素被mask，我们先把每个batch的mask_indices都扩展成(B,196,1)的形状，然后再用gather函数
    # 得到（B，147,768）
    # mask_indices[None,:,None] 把 (196,) 变成 (1,196,1)，expand 再广播成 (B,147,768) 所需的下标形状
    masked_original = torch.gather(patches, 1,
                                   mask_indices[None, :, None].expand(B, -1, patches.size(2)))
    masked_reconstructed = torch.gather(reconstructed_patches, 1,
                                   mask_indices[None, :, None].expand(B, -1, reconstructed_patches.size(2)))

    # Calculate the squared differences
    # 只对被遮住的那 147 个 patch 计算“原图 vs 重建”的平方差
    loss = (masked_original - masked_reconstructed) ** 2 # (B,147,768)

    # Calculate the mean over all dimensions
    loss = loss.mean()  # 对所有维度取均值，得到标量损失

    return loss


class MAE(nn.Module):
    """MAE 主模型：包含编码器（encoder）和解码器（decoder），完成“遮罩-重建”过程。

    参数默认值：image_size=224（图片边长），patch_size=16（每个小块边长），
    embed_dim=768（特征维度），mask_ratio=0.75（遮住 75% 的 patch）。
    因为 224/16=14，所以一张图共 14*14=196 个 patch。
    """
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # patch embbding，用于切patch
        # PatchEmbed 把 (B,3,224,224) 的图切成 196 个 patch，并各自映射成 embed_dim 维向量
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # patch 总数 = 196

        # Positional encoding
        # 位置编码：告诉模型每个 patch 在原图中的位置（Transformer 本身不知道顺序）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Encoder (sequence of blocks)
        # 编码器：12 个 Transformer Block 堆叠（每个含自注意力 + MLP）
        self.encoder = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=12, mlp_ratio=4.0) for _ in range(12)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # 编码器输出的归一化层

        # Mask token is learned during training
        # 掩码 token：一个可学习的向量，用来“占位”被遮住的 patch（代替原图信息）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Decoder
        # 解码器：把编码后的特征映射回每个 patch 的像素值（patch_size^2 * 3 即 16*16*3 个 RGB 值）
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size ** 2 * 3)  # To predict the RGB values of the masked patches
        )

    def forward(self, imgs):
        B, C, H, W = imgs.shape # (B, 3, 224, 224)

        # Extract patches
        patches = self.patch_embed(imgs) # 切 patch 并嵌入，(B,196,768)

        # Add positional encoding to patch embeddings
        x = patches + self.pos_embed.repeat(B, 1, 1) # 加上位置编码；self.pos_embed的batch是1，repeat复制成batch份

        # Masking
        N = x.shape[1] # patch的数量，196
        num_masked = int(self.mask_ratio * N)  # 需要遮住的 patch 数 = 0.75*196 = 147
        all_indices = torch.randperm(N, device=x.device) # 把196个patch的数组随机打乱，返回一个list，每个元素都是一个打乱后的下标
        mask_indices = all_indices[:num_masked]  # 选取前147个patch的下标作为被遮位置
        mask = torch.ones((N,), device=x.device) # (196)的全1向量
        mask[mask_indices] = 0  # 被遮位置置 0（此处 mask 仅作演示，后续主要用 mask_indices）

        # input unmasked tokens to the encoder
        unmasked_tokens = x.clone() # (B,196,768)，拷贝一份避免修改原张量
        unmasked_tokens[:, mask_indices] = self.mask_token # 把147个被遮patch的位置替换为可学习的mask_token
        for blk in self.encoder:  # 依次过 12 个 Transformer Block
            unmasked_tokens = blk(unmasked_tokens)
        unmasked_tokens = self.norm(unmasked_tokens) # 编码器输出归一化，(B,196,768)

        # Now we combine the masked and unmasked tokens for the decoder
        encoded_patches = unmasked_tokens

        # Add positional encodings for decoding
        encoded_patches += self.pos_embed.repeat(B, 1, 1) # 解码前再加一次位置编码，(B,196,768)

        # Decode each token to reconstruct the patches
        reconstructed_patches = self.decoder(encoded_patches) # 解码器重建每个 patch，(B,196,768)

        return reconstructed_patches, patches, mask_indices # 重建patch，原始patch，mask的下标


# Example usage:
if __name__ == '__main__':
    img = torch.rand(2, 3, 224, 224)  # Example image batch，造一个 batch=2 的随机图像做演示
    mae_model = MAE()
    # 前向传播：得到重建 patch、原始 patch 和被遮下标
    reconstructed_patches, patches, mask_indices = mae_model(img)
    # 计算重建损失（只在被遮位置）
    loss = mse_loss(reconstructed_patches, patches, mask_indices)
    print(loss)