from __future__ import annotations

from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint, count_flops_attn, gamma_embedding, normalization, zero_module


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, EmbedBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channel or channels, 3, padding=1) if use_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channel or channels, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(EmbedBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channel: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.out_channel = out_channel or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), SiLU(), nn.Conv2d(channels, self.out_channel, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channel if use_scale_shift_norm else self.out_channel),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)),
        )
        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads if num_head_channels == -1 else channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = x.shape
        flat = x.reshape(batch, channels, -1)
        h = self.attention(self.qkv(self.norm(flat)))
        return (flat + self.proj_out(h)).reshape(batch, channels, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.softmax(torch.einsum("bct,bcs->bts", q * scale, k * scale).float(), dim=-1).type(q.dtype)
        return torch.einsum("bts,bcs->bct", weight, v).reshape(batch, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.softmax(
            torch.einsum(
                "bct,bcs->bts",
                (q * scale).view(batch * self.n_heads, ch, length),
                (k * scale).view(batch * self.n_heads, ch, length),
            ).float(),
            dim=-1,
        ).type(q.dtype)
        return torch.einsum("bts,bcs->bct", weight, v.reshape(batch * self.n_heads, ch, length)).reshape(batch, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channel: int,
        inner_channel: int,
        out_channel: int,
        res_blocks: int,
        attn_res: list[int],
        dropout: float = 0.0,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.inner_channel = inner_channel
        self.dtype = torch.float16 if use_fp16 else torch.float32
        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(nn.Linear(inner_channel, cond_embed_dim), SiLU(), nn.Linear(cond_embed_dim, cond_embed_dim))

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList([EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(ch, cond_embed_dim, dropout, out_channel=int(mult * inner_channel), use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(EmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(ch, cond_embed_dim, dropout, out_channel=out_ch, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channel=out_ch)
                    )
                )
                input_block_chans.append(out_ch)
                ds *= 2

        self.middle_block = EmbedSequential(
            ResBlock(ch, cond_embed_dim, dropout, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock(ch, cond_embed_dim, dropout, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, cond_embed_dim, dropout, out_channel=int(inner_channel * mult), use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, cond_embed_dim, dropout, out_channel=out_ch, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
        self.out = nn.Sequential(normalization(ch), SiLU(), zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)))

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        hs = []
        emb = self.cond_embed(gamma_embedding(gammas.view(-1), self.inner_channel))
        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h.type(x.dtype))
