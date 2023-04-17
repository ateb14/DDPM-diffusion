import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    '''Attention block from transformer, but with a feed forward layer at the end'''

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        # channels of the input, analogous to d_model(size of the embedding) in transformer
        self.channels = channels
        # size of the input, analogous to seq_len in transformer
        self.size = size
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )  # feed forward

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        # mha args: query, key, value
        attention_val, _ = self.mha(x_ln, x_ln, x_ln)
        attention_val = attention_val + x  # residual connection
        attention_val = self.ff_self(
            attention_val) + attention_val  # residual connection
        return attention_val.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    '''conv => BN => GELU => conv => BN'''

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),  # equivalent to batchnorm
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownSample(nn.Module):
    '''Downsample => DoubleConv => DoubleConv => positional embedding'''

    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb  # positional embedding


class UpSample(nn.Module):
    '''upsample =>(residual) DoubleConv => DoubleConv => positional embedding
        in_channels here is for the conv layer after upsampling
    '''

    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # concat along channel dim
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    '''UNet with self attention blocks'''

    def __init__(self, c_in=3, c_out=3, time_dim=256, device=None):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)  # input conv

        # downsample blocks
        # input image size is 64x64, final output is 8x8 since we downsample 3 times
        self.down1 = DownSample(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = DownSample(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = DownSample(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # upsample blocks
        # we upsample 3 times symmetrically to the downsample blocks.
        # in_channels is two times the out_channels of the last block, since we concat the skip connections
        # after we upsample and before we convolve.
        self.up1 = UpSample(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = UpSample(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = UpSample(128, 64)
        self.sa6 = SelfAttention(64, 64)

        # output con
        self.outc = nn.Conv2d(64, c_out, 1)

    def pos_encoding(self, t, channels):  # same as in transformer
        inv_freq = 1. / \
            (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # downsample
        x1 = self.inc(x)  # c: 3 -> 64
        x2 = self.down1(x1, t)  # c: 64 -> 128
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)  # c: 128 -> 256
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)  # c: 256 -> 256
        x4 = self.sa3(x4)

        # bottleneck
        x4 = self.bot1(x4)  # c: 256 -> 512
        x4 = self.bot2(x4)  # c: 512 -> 512
        x4 = self.bot3(x4)  # c: 512 -> 256

        # upsample
        x = self.up1(x4, x3, t)  # x, skip_x, t， c: 256 + 256 -> 128
        x = self.sa4(x)
        x = self.up2(x, x2, t)  # c: 128 + 128 -> 64
        x = self.sa5(x)
        x = self.up3(x, x1, t)  # c: 64 + 64 -> 64
        x = self.sa6(x)

        out = self.outc(x)
        return out


if __name__ == '__main__':
    model = UNet()
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([100] * x.shape[0]).long()
    out = model(x, t)
    print(out.shape)
