from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .models import UNet
except ImportError:
    from models import UNet


def extract(a: torch.Tensor, t: torch.Tensor, x_shape=(1, 1, 1, 1)) -> torch.Tensor:
    batch, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(batch, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-6,
    linear_end: float = 1e-2,
    cosine_s: float = 8e-3,
):
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    if schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        return (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999)
    raise NotImplementedError(f"Unsupported beta schedule: {schedule}")


class GaussianConditionalDiffusion(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        target_channels: int,
        image_size: int,
        inner_channel: int,
        channel_mults: list[int],
        attn_res: list[int],
        res_blocks: int,
        dropout: float,
        beta_schedule: dict,
        num_side_classes: int | None = None,
        branch_decoder_cfg: dict | None = None,
        branch_loss_weights: dict[str, float] | None = None,
        consistency_cfg: dict | None = None,
        data_norm_cfg: dict | None = None,
    ):
        super().__init__()
        self.target_channels = target_channels
        branch_decoder_cfg = dict(branch_decoder_cfg or {})
        consistency_cfg = dict(consistency_cfg or {})
        data_norm_cfg = dict(data_norm_cfg or {})
        self.denoise_fn = UNet(
            image_size=image_size,
            in_channel=condition_channels + target_channels,
            out_channel=target_channels,
            inner_channel=inner_channel,
            channel_mults=tuple(channel_mults),
            attn_res=attn_res,
            res_blocks=res_blocks,
            dropout=dropout,
            num_head_channels=32,
            num_side_classes=num_side_classes,
            branch_decoder=bool(branch_decoder_cfg.get("enabled", False)),
            branch_split_blocks=branch_decoder_cfg.get("shared_blocks"),
        )
        self.loss_fn = nn.MSELoss()
        self.branch_loss_weights = {
            "left": float((branch_loss_weights or {}).get("left", 1.0)),
            "right": float((branch_loss_weights or {}).get("right", 1.0)),
        }
        if self.denoise_fn.branch_decoder and self.target_channels != 2:
            raise ValueError("branch_decoder currently expects target_channels=2 for left/right prediction")
        self.consistency_enabled = bool(consistency_cfg.get("enabled", False))
        if self.consistency_enabled and not self.denoise_fn.branch_decoder:
            raise ValueError("consistency_loss is currently only supported when branch_decoder is enabled")
        self.consistency_lambda = float(consistency_cfg.get("lambda", 1e-4))
        self.consistency_gamma_power = float(consistency_cfg.get("gamma_power", 1.0))
        self.consistency_beta = float(consistency_cfg.get("smooth_l1_beta", 1.0))
        self.consistency_eps = float(consistency_cfg.get("eps", 1e-6))
        self.consistency_gamma_threshold = float(consistency_cfg.get("gamma_threshold", 0.8))
        self.consistency_start_step = int(consistency_cfg.get("start_step", 0))
        self.normalize_mode = str(data_norm_cfg.get("normalize", "none"))
        self.value_range = data_norm_cfg.get("value_range")
        self.last_loss_breakdown: dict[str, float] = {}
        self.beta_schedule = beta_schedule
        self._set_noise_schedule()

    def _split_left_right(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[1] != 2:
            raise ValueError(f"Expected 2 channels for left/right split, got {tensor.shape[1]}")
        return tensor[:, 0:1], tensor[:, 1:2]

    def _compute_training_loss(self, noise_hat: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.denoise_fn.branch_decoder:
            loss = self.loss_fn(noise_hat, noise)
            return loss, {}
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
                raise ValueError("value_range is required when consistency_loss uses fixed_range_m11")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return ((tensor + 1.0) * 0.5) * (max_val - min_val) + min_val
        if self.normalize_mode == "fixed_range_01":
            if self.value_range is None:
                raise ValueError("value_range is required when consistency_loss uses fixed_range_01")
            min_val = float(self.value_range[0])
            max_val = float(self.value_range[1])
            return tensor * (max_val - min_val) + min_val
        raise ValueError(
            "consistency_loss requires an invertible normalization mode. "
            f"Got normalize={self.normalize_mode!r}; supported modes are none, fixed_range_01, fixed_range_m11."
        )

    def _compute_consistency_loss(
        self,
        condition: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        sample_gammas: torch.Tensor,
        noise_hat: torch.Tensor,
    ) -> torch.Tensor:
        y_0_hat = self.predict_start_from_noise(y_noisy, t=t, noise=noise_hat).clamp(-1.0, 1.0)
        pred_left, pred_right = self._split_left_right(y_0_hat)
        full_raw = self._denormalize_to_raw(condition)
        left_raw = self._denormalize_to_raw(pred_left)
        right_raw = self._denormalize_to_raw(pred_right)
        residual = (left_raw + right_raw - full_raw) / (full_raw + self.consistency_eps)
        consistency_mask = (sample_gammas > self.consistency_gamma_threshold).to(dtype=full_raw.dtype).view(-1, 1, 1, 1)
        consistency_map = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            beta=self.consistency_beta,
        )
        gamma_weight = sample_gammas.view(-1, 1, 1, 1).pow(self.consistency_gamma_power) * consistency_mask
        denom = gamma_weight.sum() * consistency_map.shape[1] * consistency_map.shape[2] * consistency_map.shape[3]
        if float(denom.detach().item()) <= 0.0:
            return torch.zeros((), device=condition.device, dtype=condition.dtype)
        return (gamma_weight * consistency_map).sum() / denom

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

    def p_mean_variance(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor, side_labels: torch.Tensor | None = None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, side=side_labels)
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat).clamp(-1.0, 1.0)
        return self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)

    @torch.no_grad()
    def p_sample(self, y_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor, side_labels: torch.Tensor | None = None) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(y_t=y_t, t=t, y_cond=y_cond, side_labels=side_labels)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        sample_steps: int | None = None,
        side_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = sample_steps or self.num_timesteps
        y_t = torch.randn(condition.shape[0], self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        for i in reversed(range(steps)):
            t = torch.full((condition.shape[0],), i, device=condition.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=condition, side_labels=side_labels)
        return y_t

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: torch.Tensor,
        sample_steps: int = 50,
        eta: float = 0.0,
        side_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sample_ddim_trace(condition, sample_steps=sample_steps, eta=eta, side_labels=side_labels)["final"]

    @torch.no_grad()
    def sample_ddim_trace(
        self,
        condition: torch.Tensor,
        sample_steps: int = 50,
        eta: float = 0.0,
        side_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | list[int]]:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        batch = condition.shape[0]
        y_t = torch.randn(batch, self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        time_pairs = torch.linspace(self.num_timesteps - 1, 0, sample_steps, device=condition.device).long()
        prev_pairs = torch.cat([time_pairs[1:], torch.tensor([-1], device=condition.device, dtype=torch.long)])
        samples: list[torch.Tensor] = [y_t.detach().cpu().clone()]
        predictions: list[torch.Tensor] = []
        timesteps: list[int] = [int(time_pairs[0].item())]
        for t_now, t_prev in zip(time_pairs, prev_pairs):
            t = torch.full((batch,), int(t_now.item()), device=condition.device, dtype=torch.long)
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            noise_hat = self.denoise_fn(torch.cat([condition, y_t], dim=1), noise_level, side=side_labels)
            y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat).clamp(-1.0, 1.0)
            predictions.append(y_0_hat.detach().cpu().clone())
            if int(t_prev.item()) < 0:
                y_t = y_0_hat
                samples.append(y_t.detach().cpu().clone())
                timesteps.append(-1)
                continue
            t_prev_tensor = torch.full((batch,), int(t_prev.item()), device=condition.device, dtype=torch.long)
            alpha_t = extract(self.gammas, t, y_t.shape)
            alpha_prev = extract(self.gammas, t_prev_tensor, y_t.shape)
            pred_noise = self.predict_noise_from_start(y_t, t=t, y_0=y_0_hat)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(y_t) if eta > 0 else torch.zeros_like(y_t)
            direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * pred_noise
            y_t = torch.sqrt(alpha_prev) * y_0_hat + direction + sigma * noise
            samples.append(y_t.detach().cpu().clone())
            timesteps.append(int(t_prev.item()))
        return {
            "final": y_t,
            "samples": samples,
            "predictions": predictions,
            "timesteps": timesteps,
        }

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        side_labels: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> torch.Tensor:
        batch, *_ = target.shape
        t = torch.randint(1, self.num_timesteps, (batch,), device=target.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((batch, 1), device=target.device) + gamma_t1
        noise = torch.randn_like(target)
        y_noisy = self.q_sample(target, sample_gammas.view(-1, 1, 1, 1), noise)
        noise_hat = self.denoise_fn(torch.cat([condition, y_noisy], dim=1), sample_gammas.view(-1), side=side_labels)
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
            "left_diff_loss": float(diff_breakdown["left_diff_loss"].detach().item()) if "left_diff_loss" in diff_breakdown else diff_loss_value,
            "right_diff_loss": float(diff_breakdown["right_diff_loss"].detach().item()) if "right_diff_loss" in diff_breakdown else diff_loss_value,
            "consistency_loss": float(consistency_loss.detach().item()),
            "consistency_term": consistency_term_value,
            "consistency_ratio": consistency_term_value / max(diff_loss_value, 1e-8),
            "consistency_active": 1.0 if consistency_active else 0.0,
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss
