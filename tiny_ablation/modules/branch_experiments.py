from __future__ import annotations

import math
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusion import extract, make_beta_schedule
    from models.unet import AttentionBlock, SiLU, UNet, normalization, zero_module
except ImportError:  # pragma: no cover - used when imported as a package.
    from CBCT_decouping_diffusion.diffusion import extract, make_beta_schedule
    from CBCT_decouping_diffusion.models.unet import AttentionBlock, SiLU, UNet, normalization, zero_module


def _count_parameters(module: nn.Module | None, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters() if not trainable_only or param.requires_grad)


class ConditionResidualBlock(nn.Module):
    """Small residual block used by the stronger tiny condition encoder."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = int(dilation)
        self.net = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=int(dilation)),
            normalization(channels),
            SiLU(),
            zero_module(nn.Conv2d(channels, channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SmallConditionEncoder(nn.Module):
    """Input-level condition encoder for tiny ablations.

    The legacy ``simple`` mode is the old two-convolution probe.  The default
    ``residual_multiscale`` mode is still small, but it has residual refinement
    and a down/up context path so it can extract more than raw local edges before
    the features are concatenated into the UNet input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        zero_init_output: bool = False,
        architecture: str = "residual_multiscale",
        num_blocks: int = 2,
        dilations: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.architecture = architecture
        output = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        output = zero_module(output) if zero_init_output else output
        if architecture == "simple":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                SiLU(),
                output,
            )
        elif architecture == "residual_multiscale":
            dilations = list(dilations or [1, 2, 4])
            high_blocks = [ConditionResidualBlock(hidden_channels, dilations[idx % len(dilations)]) for idx in range(max(0, int(num_blocks)))]
            low_channels = hidden_channels * 2
            low_blocks = [
                ConditionResidualBlock(low_channels, dilations[idx % len(dilations)])
                for idx in range(max(1, int(num_blocks)))
            ]
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                *high_blocks,
                _DownUpContext(hidden_channels, low_channels, low_blocks),
                normalization(hidden_channels),
                SiLU(),
                output,
            )
        else:
            raise ValueError(f"Unsupported condition encoder architecture: {architecture}")

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class _DownUpContext(nn.Module):
    def __init__(self, high_channels: int, low_channels: int, low_blocks: list[nn.Module]):
        super().__init__()
        self.down = nn.Conv2d(high_channels, low_channels, 3, stride=2, padding=1)
        self.low = nn.Sequential(*low_blocks)
        self.up = nn.Conv2d(low_channels, high_channels, 3, padding=1)
        self.fuse = nn.Conv2d(high_channels * 2, high_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low(self.down(x))
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        low = self.up(low)
        return self.fuse(torch.cat([x, low], dim=1))


class WindowAttentionBlock(nn.Module):
    """Local window self-attention drop-in replacement for UNet AttentionBlock."""

    def __init__(self, channels: int, num_heads: int = 1, window_size: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.channels = channels
        self.num_heads = max(1, num_heads)
        self.window_size = int(window_size)
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def _pad_to_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def _to_windows(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        head_dim = channels // self.num_heads
        grid_h = height // self.window_size
        grid_w = width // self.window_size
        x = x.view(batch, self.num_heads, head_dim, grid_h, self.window_size, grid_w, self.window_size)
        x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        return x.view(batch * grid_h * grid_w, self.num_heads, self.window_size * self.window_size, head_dim)

    def _from_windows(self, x: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
        channels = self.channels
        head_dim = channels // self.num_heads
        grid_h = height // self.window_size
        grid_w = width // self.window_size
        x = x.view(batch, grid_h, grid_w, self.num_heads, self.window_size, self.window_size, head_dim)
        x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return x.view(batch, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        h, pad_h, pad_w = self._pad_to_windows(self.norm(x))
        padded_h, padded_w = h.shape[-2:]
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = self._to_windows(q)
        k = self._to_windows(k)
        v = self._to_windows(v)
        scale = (self.channels // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q * scale, k.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v)
        out = self._from_windows(out, batch, padded_h, padded_w)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        return x + self.proj_out(out)


def _replace_child(module: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        module[int(name)] = child
    else:
        setattr(module, name, child)


def replace_attention_with_window(module: nn.Module, window_size: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, AttentionBlock):
            replacement = WindowAttentionBlock(
                channels=int(child.qkv.in_channels),
                num_heads=int(child.num_heads),
                window_size=window_size,
            )
            _replace_child(module, name, replacement)
            replaced += 1
        else:
            replaced += replace_attention_with_window(child, window_size)
    return replaced


class BranchConditionDenoiser(nn.Module):
    """Branch UNet wrapper with tiny-only conditioning/attention variants."""

    def __init__(
        self,
        *,
        condition_channels: int,
        target_channels: int,
        model_cfg: dict[str, Any],
        experiment_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        experiment_cfg = dict(experiment_cfg or {})
        encoder_cfg = dict(experiment_cfg.get("condition_encoder", {}))
        attention_cfg = dict(experiment_cfg.get("attention", {}))

        self.condition_encoder: SmallConditionEncoder | None = None
        self.condition_channels = int(condition_channels)
        self.target_channels = int(target_channels)
        self.retain_raw_condition = True
        encoded_channels = 0
        if encoder_cfg.get("enabled", False):
            encoded_channels = int(encoder_cfg.get("out_channels", condition_channels))
            hidden_channels = int(encoder_cfg.get("hidden_channels", max(8, encoded_channels * 2)))
            self.retain_raw_condition = bool(encoder_cfg.get("retain_concat", True))
            self.condition_encoder = SmallConditionEncoder(
                in_channels=condition_channels,
                out_channels=encoded_channels,
                hidden_channels=hidden_channels,
                zero_init_output=bool(encoder_cfg.get("zero_init_output", False)),
                architecture=str(encoder_cfg.get("architecture", "residual_multiscale")),
                num_blocks=int(encoder_cfg.get("num_blocks", 2)),
                dilations=encoder_cfg.get("dilations"),
            )
        self.encoded_channels = int(encoded_channels)

        condition_input_channels = encoded_channels
        if self.condition_encoder is None or self.retain_raw_condition:
            condition_input_channels += condition_channels
        self.condition_input_channels = int(condition_input_channels)

        branch_cfg = dict(model_cfg.get("branch_decoder", {}))
        self.unet = UNet(
            image_size=int(model_cfg["image_size"]),
            in_channel=condition_input_channels + target_channels,
            out_channel=target_channels,
            inner_channel=int(model_cfg["inner_channel"]),
            channel_mults=tuple(model_cfg["channel_mults"]),
            attn_res=list(model_cfg.get("attn_res", [])),
            res_blocks=int(model_cfg["res_blocks"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            num_head_channels=int(model_cfg.get("num_head_channels", 32)),
            branch_decoder=True,
            branch_split_blocks=branch_cfg.get("shared_blocks"),
        )

        self.attention_type = str(attention_cfg.get("type", "vanilla"))
        self.window_attention_blocks = 0
        if self.attention_type == "window":
            self.window_attention_blocks = replace_attention_with_window(
                self.unet,
                window_size=int(attention_cfg.get("window_size", 8)),
            )
        elif self.attention_type != "vanilla":
            raise ValueError(f"Unsupported tiny attention type: {self.attention_type}")

    @property
    def branch_decoder(self) -> bool:
        return bool(self.unet.branch_decoder)

    def _build_condition_input(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_encoder is None:
            return condition
        encoded = self.condition_encoder(condition)
        if self.retain_raw_condition:
            return torch.cat([condition, encoded], dim=1)
        return encoded

    def forward(
        self,
        condition: torch.Tensor,
        y_noisy: torch.Tensor,
        gammas: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_input = torch.cat([self._build_condition_input(condition), y_noisy], dim=1)
        return self.unet(model_input, gammas, side=side)

    def parameter_summary(self) -> dict[str, int]:
        attention_params = 0
        for module in self.unet.modules():
            if isinstance(module, (AttentionBlock, WindowAttentionBlock)):
                attention_params += _count_parameters(module)
        return {
            "denoiser_parameters": _count_parameters(self),
            "condition_encoder_parameters": _count_parameters(self.condition_encoder),
            "unet_parameters": _count_parameters(self.unet),
            "attention_parameters": attention_params,
            "condition_input_channels": self.condition_input_channels,
            "encoded_condition_channels": self.encoded_channels,
        }


class TinyBranchDiffusion(nn.Module):
    """Minimal branch diffusion used only by tiny ablation experiments."""

    def __init__(
        self,
        *,
        condition_channels: int,
        target_channels: int,
        model_cfg: dict[str, Any],
        beta_schedule: dict[str, Any],
        branch_loss_weights: dict[str, float] | None = None,
        consistency_cfg: dict[str, Any] | None = None,
        data_norm_cfg: dict[str, Any] | None = None,
        experiment_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        if target_channels != 2:
            raise ValueError("TinyBranchDiffusion expects dual left/right targets.")
        self.target_channels = target_channels
        self.denoise_fn = BranchConditionDenoiser(
            condition_channels=condition_channels,
            target_channels=target_channels,
            model_cfg=model_cfg,
            experiment_cfg=experiment_cfg,
        )
        self.loss_fn = nn.MSELoss()
        self.branch_loss_weights = {
            "left": float((branch_loss_weights or {}).get("left", 1.0)),
            "right": float((branch_loss_weights or {}).get("right", 1.0)),
        }
        consistency_cfg = dict(consistency_cfg or {})
        data_norm_cfg = dict(data_norm_cfg or {})
        self.consistency_enabled = bool(consistency_cfg.get("enabled", False))
        self.consistency_lambda = float(consistency_cfg.get("lambda", 1e-4))
        self.consistency_gamma_power = float(consistency_cfg.get("gamma_power", 1.0))
        self.consistency_beta = float(consistency_cfg.get("smooth_l1_beta", 1.0))
        self.consistency_eps = float(consistency_cfg.get("eps", 1e-6))
        self.consistency_gamma_threshold = float(consistency_cfg.get("gamma_threshold", 0.8))
        self.consistency_start_step = int(consistency_cfg.get("start_step", 0))
        self.consistency_denominator = str(consistency_cfg.get("denominator", "mean_abs_full"))
        self.normalize_mode = str(data_norm_cfg.get("normalize", "none"))
        self.value_range = data_norm_cfg.get("value_range")
        self.last_loss_breakdown: dict[str, float] = {}
        self.beta_schedule = beta_schedule
        self._set_noise_schedule()

    def _set_noise_schedule(self) -> None:
        device = next(self.denoise_fn.parameters()).device
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1)))
        self.register_buffer("sqrt_gammas", to_torch(np.sqrt(gammas)))
        self.register_buffer("sqrt_one_minus_gammas", to_torch(np.sqrt(1.0 - gammas)))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)))

    def _split_left_right(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor[:, 0:1], tensor[:, 1:2]

    def _compute_training_loss(self, noise_hat: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_left, pred_right = self._split_left_right(noise_hat)
        noise_left, noise_right = self._split_left_right(noise)
        left_loss = self.loss_fn(pred_left, noise_left)
        right_loss = self.loss_fn(pred_right, noise_right)
        left_weight = self.branch_loss_weights["left"]
        right_weight = self.branch_loss_weights["right"]
        total_loss = (left_weight * left_loss + right_weight * right_loss) / max(left_weight + right_weight, 1e-8)
        return total_loss, {
            "left_diff_loss": left_loss,
            "right_diff_loss": right_loss,
        }

    def _denormalize_to_raw(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "none":
            return tensor
        if self.normalize_mode == "fixed_range_m11":
            if self.value_range is None:
                raise ValueError("value_range is required for fixed_range_m11 consistency.")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return ((tensor + 1.0) * 0.5) * (max_val - min_val) + min_val
        if self.normalize_mode == "fixed_range_01":
            if self.value_range is None:
                raise ValueError("value_range is required for fixed_range_01 consistency.")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return tensor * (max_val - min_val) + min_val
        raise ValueError(f"Unsupported normalize mode for consistency: {self.normalize_mode}")

    def _consistency_scale(self, full_raw: torch.Tensor) -> torch.Tensor:
        mode = self.consistency_denominator
        if mode == "full_pixel":
            return full_raw.abs().clamp_min(self.consistency_eps)
        if mode == "mean_abs_full":
            return full_raw.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.consistency_eps)
        if mode == "value_range":
            if self.value_range is None:
                raise ValueError("value_range is required for consistency denominator=value_range.")
            return torch.full_like(full_raw, max(float(self.value_range[1]) - float(self.value_range[0]), self.consistency_eps))
        if mode == "none":
            return torch.ones_like(full_raw)
        raise ValueError(f"Unsupported consistency denominator: {mode}")

    def _raw_consistency_residual(self, condition: torch.Tensor, dual_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = self._split_left_right(dual_tensor)
        full_raw = self._denormalize_to_raw(condition)
        left_raw = self._denormalize_to_raw(left)
        right_raw = self._denormalize_to_raw(right)
        return left_raw + right_raw - full_raw, full_raw

    @torch.no_grad()
    def consistency_metrics(self, condition: torch.Tensor, dual_tensor: torch.Tensor, prefix: str) -> dict[str, float]:
        residual, full_raw = self._raw_consistency_residual(condition, dual_tensor)
        abs_residual = residual.abs()
        full_abs_mean = full_raw.abs().mean().clamp_min(self.consistency_eps)
        full_mean = full_raw.mean()
        sum_mean = (full_raw + residual).mean()
        return {
            f"{prefix}_consistency_mae": float(abs_residual.mean().detach().item()),
            f"{prefix}_consistency_nmae": float((abs_residual.mean() / full_abs_mean).detach().item()),
            f"{prefix}_consistency_signed_mean": float(residual.mean().detach().item()),
            f"{prefix}_sum_over_full_mean_ratio": float((sum_mean / full_mean.clamp_min(self.consistency_eps)).detach().item()),
        }

    def _compute_consistency_loss(
        self,
        condition: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        sample_gammas: torch.Tensor,
        noise_hat: torch.Tensor,
    ) -> torch.Tensor:
        y_0_hat = self.predict_start_from_noise(y_noisy, t=t, noise=noise_hat).clamp(-1.0, 1.0)
        residual_raw, full_raw = self._raw_consistency_residual(condition, y_0_hat)
        residual = residual_raw / self._consistency_scale(full_raw)
        mask = (sample_gammas > self.consistency_gamma_threshold).to(dtype=full_raw.dtype).view(-1, 1, 1, 1)
        consistency_map = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            beta=self.consistency_beta,
        )
        gamma_weight = sample_gammas.view(-1, 1, 1, 1).pow(self.consistency_gamma_power) * mask
        denom = gamma_weight.sum() * consistency_map.shape[1] * consistency_map.shape[2] * consistency_map.shape[3]
        if float(denom.detach().item()) <= 0.0:
            return torch.zeros((), device=condition.device, dtype=condition.dtype)
        return (gamma_weight * consistency_map).sum() / denom

    def q_sample(self, y_0: torch.Tensor, sample_gammas: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def predict_start_from_noise(self, y_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise

    def predict_noise_from_start(self, y_t: torch.Tensor, t: torch.Tensor, y_0: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - y_0) / extract(self.sqrt_recipm1_gammas, t, y_t.shape)

    def q_posterior(self, y_0_hat: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance

    def p_mean_variance(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        noise_hat = self.denoise_fn(y_cond, y_t, noise_level)
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat).clamp(-1.0, 1.0)
        return self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)

    @torch.no_grad()
    def p_sample(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(y_t=y_t, t=t, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, sample_steps: int | None = None) -> torch.Tensor:
        steps = sample_steps or self.num_timesteps
        y_t = torch.randn(condition.shape[0], self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        for i in reversed(range(steps)):
            t = torch.full((condition.shape[0],), i, device=condition.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=condition)
        return y_t

    @torch.no_grad()
    def sample_ddim(self, condition: torch.Tensor, sample_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        batch = condition.shape[0]
        y_t = torch.randn(batch, self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        time_pairs = torch.linspace(self.num_timesteps - 1, 0, sample_steps, device=condition.device).long()
        prev_pairs = torch.cat([time_pairs[1:], torch.tensor([-1], device=condition.device, dtype=torch.long)])
        for t_now, t_prev in zip(time_pairs, prev_pairs):
            t = torch.full((batch,), int(t_now.item()), device=condition.device, dtype=torch.long)
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            noise_hat = self.denoise_fn(condition, y_t, noise_level)
            y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat).clamp(-1.0, 1.0)
            if int(t_prev.item()) < 0:
                y_t = y_0_hat
                continue
            t_prev_tensor = torch.full((batch,), int(t_prev.item()), device=condition.device, dtype=torch.long)
            alpha_t = extract(self.gammas, t, y_t.shape)
            alpha_prev = extract(self.gammas, t_prev_tensor, y_t.shape)
            pred_noise = self.predict_noise_from_start(y_t, t=t, y_0=y_0_hat)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(y_t) if eta > 0 else torch.zeros_like(y_t)
            direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * pred_noise
            y_t = torch.sqrt(alpha_prev) * y_0_hat + direction + sigma * noise
        return y_t

    def forward(self, target: torch.Tensor, condition: torch.Tensor, global_step: int | None = None) -> torch.Tensor:
        batch, *_ = target.shape
        t = torch.randint(1, self.num_timesteps, (batch,), device=target.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((batch, 1), device=target.device) + gamma_t1
        noise = torch.randn_like(target)
        y_noisy = self.q_sample(target, sample_gammas.view(-1, 1, 1, 1), noise)
        noise_hat = self.denoise_fn(condition, y_noisy, sample_gammas.view(-1))
        diff_loss, diff_breakdown = self._compute_training_loss(noise_hat, noise)
        consistency_loss = torch.zeros((), device=target.device, dtype=diff_loss.dtype)
        consistency_active = self.consistency_enabled and (global_step is None or global_step >= self.consistency_start_step)
        if consistency_active:
            consistency_loss = self._compute_consistency_loss(condition, y_noisy, t, sample_gammas, noise_hat)
        consistency_term = self.consistency_lambda * consistency_loss
        total_loss = diff_loss + consistency_term
        diff_loss_value = float(diff_loss.detach().item())
        consistency_term_value = float(consistency_term.detach().item())
        self.last_loss_breakdown = {
            "diff_loss": diff_loss_value,
            "left_diff_loss": float(diff_breakdown["left_diff_loss"].detach().item()),
            "right_diff_loss": float(diff_breakdown["right_diff_loss"].detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "consistency_term": consistency_term_value,
            "consistency_ratio": consistency_term_value / max(diff_loss_value, 1e-8),
            "consistency_active": 1.0 if consistency_active else 0.0,
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss

    def parameter_summary(self) -> dict[str, int]:
        summary = self.denoise_fn.parameter_summary()
        summary.update(
            {
                "model_parameters": _count_parameters(self),
                "trainable_parameters": _count_parameters(self, trainable_only=True),
            }
        )
        return summary
