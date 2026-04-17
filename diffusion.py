from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .models import UNet


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
    ):
        super().__init__()
        self.target_channels = target_channels
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
        )
        self.loss_fn = nn.MSELoss()
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
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
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
            noise_hat = self.denoise_fn(torch.cat([condition, y_t], dim=1), noise_level)
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

    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, *_ = target.shape
        t = torch.randint(1, self.num_timesteps, (batch,), device=target.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((batch, 1), device=target.device) + gamma_t1
        noise = torch.randn_like(target)
        y_noisy = self.q_sample(target, sample_gammas.view(-1, 1, 1, 1), noise)
        noise_hat = self.denoise_fn(torch.cat([condition, y_noisy], dim=1), sample_gammas.view(-1))
        return self.loss_fn(noise_hat, noise)
