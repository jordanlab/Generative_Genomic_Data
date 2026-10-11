"""Opt-in conditional WGAN-GP model variants.

Two variants are provided:

* FiLMConditionalWGAN: PC/vector-space WGAN where ancestry conditions modulate every
  hidden layer with FiLM rather than being concatenated only at the input.
* GenotypeConvWGAN: direct SNP-space residual 1-D convolutional WGAN.  It learns
  contiguous-locus features directly instead of relying on PCA reconstruction.

Both trainers optionally support population-balanced sampling and PacGAN-style
critic packing.  They intentionally expose the same ``fit/sample/save_checkpoint``
surface as the baseline model in conditional_wgan_pipeline.py.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# -----------------------------------------------------------------------------
# Shared layers / helpers
# -----------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise linear modulation: h -> gamma(c) * h + beta(c)."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.affine = nn.Linear(cond_dim, 2 * feature_dim)
        # Identity at initialization: gamma=1, beta=0.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma_delta, beta = self.affine(c).chunk(2, dim=1)
        gamma = 1.0 + gamma_delta
        if h.ndim == 2:
            return gamma * h + beta
        if h.ndim == 3:
            return gamma[:, :, None] * h + beta[:, :, None]
        raise ValueError(f"FiLM expects rank-2 or rank-3 features; got {h.shape}.")


def _parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _pack_vector(x: torch.Tensor, c: torch.Tensor, pack_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if pack_size == 1:
        return x, c
    n = (x.shape[0] // pack_size) * pack_size
    if n == 0:
        raise ValueError("Batch is smaller than pack_size.")
    x = x[:n].reshape(n // pack_size, -1)
    c = c[:n].reshape(n // pack_size, -1)
    return x, c


def _pack_sequence(x: torch.Tensor, c: torch.Tensor, pack_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input x: (B, L); packed critic input: (B/P, P, L), i.e. samples become channels.
    n = (x.shape[0] // pack_size) * pack_size
    if n == 0:
        raise ValueError("Batch is smaller than pack_size.")
    x = x[:n].reshape(n // pack_size, pack_size, x.shape[1])
    c = c[:n].reshape(n // pack_size, -1)
    return x, c


def _balanced_sampler(labels: Sequence[str], seed: int) -> WeightedRandomSampler:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("group labels must be one-dimensional.")
    unique, counts = np.unique(labels, return_counts=True)
    inv = {u: 1.0 / n for u, n in zip(unique, counts)}
    weights = np.asarray([inv[x] for x in labels], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


@dataclass
class TrainerConfig:
    latent_dim: int = 32
    cond_dim: int = 3
    n_epochs: int = 3000
    batch_size: int = 64
    lr: float = 1e-4
    n_critic: int = 5
    lambda_gp: float = 10.0
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    val_fraction: float = 0.2
    device: str = "auto"
    seed: int = 42
    log_every: int = 500
    pack_size: int = 1
    balanced_sampling: bool = False


class _BaseWGANGP:
    """Shared WGAN-GP optimizer loop for vector and sequence variants."""

    model_kind = "base"

    def __init__(self, config: TrainerConfig):
        self.config = config
        self._device = None
        self._G = None
        self._C = None
        self.history = None
        self.train_indices = None
        self.val_indices = None
        self._best_state = None
        self._best_epoch = None
        self._best_score = None
        self._baseline_state = None
        self._first_state = None
        self._last_state = None

    def _resolve_device(self) -> torch.device:
        if self.config.device and self.config.device.lower() != "auto":
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_networks(self, input_dim: int) -> Tuple[nn.Module, nn.Module]:
        raise NotImplementedError

    def _prepare_data(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _restore_samples(self, generated: np.ndarray) -> np.ndarray:
        return generated

    def _pack(self, x: torch.Tensor, c: torch.Tensor):
        raise NotImplementedError

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alpha_shape = [real.shape[0]] + [1] * (real.ndim - 1)
        alpha = torch.rand(alpha_shape, device=self._device)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        scores = self._C(interp, cond)
        grads = torch.autograd.grad(
            outputs=scores,
            inputs=interp,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        flat = grads.reshape(grads.shape[0], -1)
        return ((flat.norm(2, dim=1) - 1.0) ** 2).mean()

    def _eval_losses(self, loader):
        if loader is None:
            return {"g": np.nan, "c_total": np.nan, "c_wasserstein": np.nan, "gp": np.nan}
        self._G.eval(); self._C.eval()
        sums = {"g": 0.0, "c_total": 0.0, "c_wasserstein": 0.0, "gp": 0.0}
        n_batches = 0
        for real_base, cond_base in loader:
            real_base = real_base.to(self._device)
            cond_base = cond_base.to(self._device)
            n = (real_base.shape[0] // self.config.pack_size) * self.config.pack_size
            if n == 0:
                continue
            real_base, cond_base = real_base[:n], cond_base[:n]
            z = torch.randn(n, self.config.latent_dim, device=self._device)
            with torch.no_grad():
                fake_base = self._G(z, cond_base)
            real, cond = self._pack(real_base, cond_base)
            fake, _ = self._pack(fake_base, cond_base)
            with torch.no_grad():
                c_fake = self._C(fake, cond).mean()
                c_real = self._C(real, cond).mean()
                g = -c_fake
                w = c_fake - c_real
            gp = self._gradient_penalty(real, fake.detach(), cond)
            total = w + self.config.lambda_gp * gp
            sums["g"] += float(g.item())
            sums["c_total"] += float(total.item())
            sums["c_wasserstein"] += float(w.item())
            sums["gp"] += float(gp.item())
            n_batches += 1
        self._G.train(); self._C.train()
        if n_batches == 0:
            return {k: np.nan for k in sums}
        return {k: v / n_batches for k, v in sums.items()}

    def fit(
        self,
        x: np.ndarray,
        conditions: np.ndarray,
        group_labels: Optional[Sequence[str]] = None,
        epoch_callback=None,
    ) -> Dict[str, list]:
        cfg = self.config
        if not (0 <= cfg.val_fraction < 1):
            raise ValueError("val_fraction must lie in [0,1).")
        if cfg.pack_size < 1:
            raise ValueError("pack_size must be >=1.")
        if cfg.batch_size < cfg.pack_size:
            raise ValueError("batch_size must be >= pack_size.")
        if conditions.shape != (len(x), cfg.cond_dim):
            raise ValueError(
                f"conditions must have shape ({len(x)},{cfg.cond_dim}); got {conditions.shape}."
            )

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._device = self._resolve_device()
        x_prepared = self._prepare_data(np.asarray(x, dtype=np.float32))
        cond = np.asarray(conditions, dtype=np.float32)
        n_samples, input_dim = x_prepared.shape

        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n_samples)
        n_val = int(round(n_samples * cfg.val_fraction))
        if cfg.val_fraction > 0 and n_samples >= 5:
            n_val = max(1, min(n_val, n_samples - 2))
        else:
            n_val = 0
        self.val_indices = perm[:n_val]
        self.train_indices = perm[n_val:]

        x_train = torch.from_numpy(x_prepared[self.train_indices])
        c_train = torch.from_numpy(cond[self.train_indices])
        train_ds = TensorDataset(x_train, c_train)
        sampler = None
        shuffle = True
        if cfg.balanced_sampling:
            if group_labels is None:
                raise ValueError("balanced_sampling=True requires group_labels.")
            group_labels = np.asarray(group_labels)
            sampler = _balanced_sampler(group_labels[self.train_indices], cfg.seed)
            shuffle = False
        train_loader = DataLoader(
            train_ds,
            batch_size=min(cfg.batch_size, len(train_ds)),
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            drop_last=(len(train_ds) > cfg.batch_size),
            generator=torch.Generator().manual_seed(cfg.seed) if sampler is None else None,
        )

        val_loader = None
        if n_val:
            val_ds = TensorDataset(
                torch.from_numpy(x_prepared[self.val_indices]),
                torch.from_numpy(cond[self.val_indices]),
            )
            val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, n_val), shuffle=False)

        self._G, self._C = self._build_networks(input_dim)
        self._G = self._G.to(self._device)
        self._C = self._C.to(self._device)
        # Randomly initialized weights before any optimizer update: semantic baseline (epoch 0).
        self._baseline_state = {
            "G": copy.deepcopy(self._G.state_dict()),
            "C": copy.deepcopy(self._C.state_dict()),
        }
        opt_g = torch.optim.Adam(
            self._G.parameters(), lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad,
        )
        opt_c = torch.optim.Adam(
            self._C.parameters(), lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad,
        )

        print(f"  model kind      : {self.model_kind}", flush=True)
        print(f"  device          : {self._device}", flush=True)
        print(f"  samples/features: {n_samples} x {input_dim}", flush=True)
        print(f"  split           : train={len(self.train_indices)} val={n_val}", flush=True)
        print(f"  balanced sampler: {cfg.balanced_sampling}", flush=True)
        print(f"  PacGAN pack size: {cfg.pack_size}", flush=True)
        print("\n  Generator architecture:\n" + str(self._G), flush=True)
        print(f"  Generator parameters: {_parameter_count(self._G):,}", flush=True)
        print("\n  Critic architecture:\n" + str(self._C), flush=True)
        print(f"  Critic parameters: {_parameter_count(self._C):,}", flush=True)
        print(f"\n  Optimizer G: {opt_g}", flush=True)
        print(f"  Optimizer C: {opt_c}", flush=True)
        print("  Loss G: -E[C(G(z,c),c)]", flush=True)
        print("  Loss C: E[C(fake,c)] - E[C(real,c)] + lambda_gp * GP", flush=True)
        print("  GP: E[(||grad_xhat C(xhat,c)||_2 - 1)^2]", flush=True)

        hist = {
            "epoch": [], "train_g": [], "val_g": [],
            "train_c_total": [], "val_c_total": [],
            "train_c_wasserstein": [], "val_c_wasserstein": [],
            "train_gp": [], "val_gp": [],
        }
        t0 = time.time()
        self._G.train(); self._C.train()
        for epoch in range(1, cfg.n_epochs + 1):
            sums = {"g": 0.0, "c_total": 0.0, "c_wasserstein": 0.0, "gp": 0.0}
            ng = nc = 0
            for real_base, cond_base in train_loader:
                real_base = real_base.to(self._device)
                cond_base = cond_base.to(self._device)
                n = (real_base.shape[0] // cfg.pack_size) * cfg.pack_size
                if n == 0:
                    continue
                real_base, cond_base = real_base[:n], cond_base[:n]

                for _ in range(cfg.n_critic):
                    z = torch.randn(n, cfg.latent_dim, device=self._device)
                    fake_base = self._G(z, cond_base).detach()
                    real, cond_packed = self._pack(real_base, cond_base)
                    fake, _ = self._pack(fake_base, cond_base)
                    gp = self._gradient_penalty(real, fake, cond_packed)
                    c_fake = self._C(fake, cond_packed).mean()
                    c_real = self._C(real, cond_packed).mean()
                    w = c_fake - c_real
                    c_total = w + cfg.lambda_gp * gp
                    opt_c.zero_grad(set_to_none=True)
                    c_total.backward()
                    opt_c.step()
                    sums["c_total"] += float(c_total.item())
                    sums["c_wasserstein"] += float(w.item())
                    sums["gp"] += float(gp.item())
                    nc += 1

                z = torch.randn(n, cfg.latent_dim, device=self._device)
                fake_base = self._G(z, cond_base)
                fake, cond_packed = self._pack(fake_base, cond_base)
                g = -self._C(fake, cond_packed).mean()
                opt_g.zero_grad(set_to_none=True)
                g.backward()
                opt_g.step()
                sums["g"] += float(g.item())
                ng += 1

            tr = {
                "g": sums["g"] / max(ng, 1),
                "c_total": sums["c_total"] / max(nc, 1),
                "c_wasserstein": sums["c_wasserstein"] / max(nc, 1),
                "gp": sums["gp"] / max(nc, 1),
            }
            va = self._eval_losses(val_loader)
            hist["epoch"].append(epoch)
            for key in ("g", "c_total", "c_wasserstein", "gp"):
                hist[f"train_{key}"].append(tr[key])
                hist[f"val_{key}"].append(va[key])

            callback_metrics = None
            if epoch_callback is not None:
                callback_metrics = epoch_callback(self, epoch)
                if callback_metrics and "selection_score" in callback_metrics:
                    score = float(callback_metrics["selection_score"])
                    if np.isfinite(score) and (self._best_score is None or score < self._best_score):
                        self._best_score = score
                        self._best_epoch = epoch
                        self._best_state = {
                            "G": copy.deepcopy(self._G.state_dict()),
                            "C": copy.deepcopy(self._C.state_dict()),
                        }
                # Sampling inside a callback switches G to eval mode. Restore training mode.
                self._G.train(); self._C.train()

            if epoch == 1:
                self._first_state = {
                    "G": copy.deepcopy(self._G.state_dict()),
                    "C": copy.deepcopy(self._C.state_dict()),
                }

            if cfg.log_every > 0 and (epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.n_epochs):
                msg = (
                    f"  [epoch {epoch:>5}/{cfg.n_epochs}] "
                    f"train G={tr['g']:+.4f} C={tr['c_total']:+.4f} "
                    f"W={tr['c_wasserstein']:+.4f} GP={tr['gp']:.4f}"
                )
                if val_loader is not None:
                    msg += (
                        f" | val G={va['g']:+.4f} C={va['c_total']:+.4f} "
                        f"W={va['c_wasserstein']:+.4f} GP={va['gp']:.4f}"
                    )
                if callback_metrics and "selection_score" in callback_metrics:
                    msg += f" | multi_metric={callback_metrics['selection_score']:.4f}"
                msg += f" [{time.time() - t0:.1f}s]"
                print(msg, flush=True)

        # Preserve the final-epoch weights before optionally restoring the best validation checkpoint.
        self._last_state = {
            "G": copy.deepcopy(self._G.state_dict()),
            "C": copy.deepcopy(self._C.state_dict()),
        }
        if self._best_state is not None:
            self._G.load_state_dict(self._best_state["G"])
            self._C.load_state_dict(self._best_state["C"])
            print(
                f"  Restored best multi-metric checkpoint from epoch {self._best_epoch} "
                f"(score={self._best_score:.6g}).",
                flush=True,
            )
        self._G.eval(); self._C.eval()
        self.history = hist
        return hist

    def sample(self, n_samples: int, conditions: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        if self._G is None:
            raise RuntimeError("Call fit() before sample().")
        c = np.asarray(conditions, dtype=np.float32)
        if c.ndim == 1:
            c = np.tile(c, (n_samples, 1))
        if c.shape != (n_samples, self.config.cond_dim):
            raise ValueError(f"conditions must be ({n_samples},{self.config.cond_dim}); got {c.shape}.")
        if z is None:
            z_t = torch.randn(n_samples, self.config.latent_dim, device=self._device)
        else:
            z = np.asarray(z, dtype=np.float32)
            if z.shape != (n_samples, self.config.latent_dim):
                raise ValueError("z has the wrong shape.")
            z_t = torch.from_numpy(z).to(self._device)
        c_t = torch.from_numpy(c).to(self._device)
        self._G.eval()
        with torch.no_grad():
            out = self._G(z_t, c_t).cpu().numpy()
        return self._restore_samples(out).astype(np.float32)

    def _state_dicts_for(self, state: str):
        if self._G is None:
            raise RuntimeError("No trained model state is available.")
        if state == "current":
            return {"G": self._G.state_dict(), "C": self._C.state_dict()}
        if state == "baseline":
            if self._baseline_state is None:
                raise RuntimeError("Baseline pre-training state is unavailable; fit/initialize the model first.")
            return self._baseline_state
        if state == "first":
            if self._first_state is None:
                raise RuntimeError("First-epoch state is unavailable; fit the model first.")
            return self._first_state
        if state == "last":
            if self._last_state is None:
                raise RuntimeError("Final-epoch state is unavailable; fit the model first.")
            return self._last_state
        if state == "best":
            if self._best_state is None:
                raise RuntimeError("Best-validation state is unavailable; enable validation checkpointing.")
            return self._best_state
        raise ValueError("state must be one of: current, baseline, first, best, last")

    def activate_checkpoint_state(self, state: str):
        """Load one captured epoch state into the networks for sampling/evaluation."""
        state_dicts = self._state_dicts_for(state)
        self._G.load_state_dict(state_dicts["G"])
        self._C.load_state_dict(state_dicts["C"])
        self._G.eval(); self._C.eval()

    def save_checkpoint(self, path: str, state: str = "current"):
        """Save current, baseline-, first-, best-validation-, or final-epoch weights."""
        state_dicts = self._state_dicts_for(state)
        torch.save(
            {
                "model_kind": self.model_kind,
                "checkpoint_state": state,
                "config": asdict(self.config),
                "generator_state_dict": state_dicts["G"],
                "critic_state_dict": state_dicts["C"],
                "train_indices": self.train_indices,
                "val_indices": self.val_indices,
                "baseline_epoch": 0 if self._baseline_state is not None else None,
                "first_epoch": 1 if self._first_state is not None else None,
                "best_epoch": self._best_epoch,
                "best_score": self._best_score,
                "last_epoch": self.config.n_epochs if self._last_state is not None else None,
                "extra_state": self._extra_checkpoint_state(),
            },
            path,
        )
        print(f"  Model checkpoint saved ({state}): {path}", flush=True)

    def _extra_checkpoint_state(self):
        return {}


# -----------------------------------------------------------------------------
# FiLM MLP in PCA/vector space
# -----------------------------------------------------------------------------

class _FiLMMLPGenerator(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dims, output_dim, negative_slope=0.2, eps=1e-5):
        super().__init__()
        dims = [latent_dim] + list(hidden_dims)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(h, eps=eps) for h in hidden_dims])
        self.films = nn.ModuleList([FiLM(cond_dim, h) for h in hidden_dims])
        self.out = nn.Linear(hidden_dims[-1], output_dim)
        self.negative_slope = negative_slope

    def forward(self, z, c):
        h = z
        for linear, norm, film in zip(self.layers, self.norms, self.films):
            h = F.leaky_relu(film(norm(linear(h)), c), self.negative_slope)
        return self.out(h)


class _FiLMMLPCritic(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims, negative_slope=0.2):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.films = nn.ModuleList([FiLM(cond_dim, h) for h in hidden_dims])
        self.out = nn.Linear(hidden_dims[-1], 1)
        self.negative_slope = negative_slope

    def forward(self, x, c):
        h = x
        for linear, film in zip(self.layers, self.films):
            h = F.leaky_relu(film(linear(h), c), self.negative_slope)
        return self.out(h)


class FiLMConditionalWGAN(_BaseWGANGP):
    """PCA/vector WGAN with repeated feature-wise ancestry conditioning."""

    model_kind = "pca_film_mlp"

    def __init__(
        self,
        latent_dim=32,
        cond_dim=3,
        hidden_dims=None,
        n_epochs=3000,
        batch_size=64,
        lr=1e-4,
        n_critic=5,
        lambda_gp=10.0,
        adam_beta1=0.0,
        adam_beta2=0.9,
        adam_eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        leaky_relu_slope=0.2,
        layernorm_eps=1e-5,
        norm_eps=1e-8,
        val_fraction=0.2,
        device="auto",
        seed=42,
        log_every=500,
        pack_size=1,
        balanced_sampling=False,
    ):
        super().__init__(TrainerConfig(
            latent_dim=latent_dim, cond_dim=cond_dim, n_epochs=n_epochs,
            batch_size=batch_size, lr=lr, n_critic=n_critic, lambda_gp=lambda_gp,
            adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
            weight_decay=weight_decay, amsgrad=amsgrad, val_fraction=val_fraction,
            device=device, seed=seed, log_every=log_every, pack_size=pack_size,
            balanced_sampling=balanced_sampling,
        ))
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.leaky_relu_slope = leaky_relu_slope
        self.layernorm_eps = layernorm_eps
        self.norm_eps = norm_eps
        self._mean = None
        self._std = None

    def _prepare_data(self, x):
        self._mean = x.mean(axis=0).astype(np.float32)
        self._std = (x.std(axis=0) + self.norm_eps).astype(np.float32)
        return ((x - self._mean) / self._std).astype(np.float32)

    def _restore_samples(self, generated):
        return generated * self._std + self._mean

    def _build_networks(self, input_dim):
        g = _FiLMMLPGenerator(
            self.config.latent_dim, self.config.cond_dim, self.hidden_dims, input_dim,
            self.leaky_relu_slope, self.layernorm_eps,
        )
        c = _FiLMMLPCritic(
            input_dim * self.config.pack_size,
            self.config.cond_dim * self.config.pack_size,
            self.hidden_dims,
            self.leaky_relu_slope,
        )
        return g, c

    def _pack(self, x, c):
        return _pack_vector(x, c, self.config.pack_size)

    def _extra_checkpoint_state(self):
        return {
            "mean": self._mean,
            "std": self._std,
            "hidden_dims": self.hidden_dims,
            "leaky_relu_slope": self.leaky_relu_slope,
            "layernorm_eps": self.layernorm_eps,
            "norm_eps": self.norm_eps,
        }


# -----------------------------------------------------------------------------
# Direct genotype 1-D convolutional WGAN
# -----------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, down=False, up=False, negative_slope=0.2):
        super().__init__()
        if down and up:
            raise ValueError("A residual block cannot both upsample and downsample.")
        self.down = down
        self.up = up
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.film1 = FiLM(cond_dim, out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.negative_slope = negative_slope

    def _resize(self, x):
        if self.up:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return x

    def forward(self, x, c):
        h = self._resize(x)
        skip = self.skip(h)
        h = self.conv1(h)
        h = F.leaky_relu(self.film1(h, c), self.negative_slope)
        h = self.conv2(h)
        h = self.film2(h, c)
        if self.down:
            h = F.avg_pool1d(h, kernel_size=2, stride=2, ceil_mode=True)
            skip = F.avg_pool1d(skip, kernel_size=2, stride=2, ceil_mode=True)
        return F.leaky_relu((h + skip) / math.sqrt(2.0), self.negative_slope)


class _ConvGenerator(nn.Module):
    def __init__(self, n_snps, latent_dim, cond_dim, channels, n_blocks, negative_slope=0.2, position_bias=True):
        super().__init__()
        self.n_snps = int(n_snps)
        self.n_blocks = int(n_blocks)
        self.base_len = max(4, math.ceil(self.n_snps / (2 ** self.n_blocks)))
        self.fc = nn.Linear(latent_dim, channels * self.base_len)
        blocks = []
        ch = channels
        for _ in range(self.n_blocks):
            out_ch = max(16, ch // 2)
            blocks.append(_ResBlock1D(ch, out_ch, cond_dim, up=True, negative_slope=negative_slope))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.to_genotype = nn.Conv1d(ch, 1, kernel_size=5, padding=2)
        self.position_bias = nn.Parameter(torch.zeros(1, 1, self.n_snps)) if position_bias else None

    def forward(self, z, c):
        h = self.fc(z).reshape(z.shape[0], -1, self.base_len)
        for block in self.blocks:
            h = block(h, c)
        if h.shape[-1] != self.n_snps:
            h = F.interpolate(h, size=self.n_snps, mode="linear", align_corners=False)
        logits = self.to_genotype(h)
        if self.position_bias is not None:
            logits = logits + self.position_bias
        # Continuous dosage on the centered scale: -1,0,+1 correspond to 0,1,2 dosages.
        return torch.tanh(logits).squeeze(1)


class _ConvCritic(nn.Module):
    def __init__(self, n_snps, cond_dim, pack_size, channels, n_blocks, negative_slope=0.2, position_channel=True):
        super().__init__()
        self.n_snps = int(n_snps)
        self.position_channel = position_channel
        in_ch = pack_size + (1 if position_channel else 0)
        blocks = []
        ch = channels
        self.stem = nn.Conv1d(in_ch, ch, kernel_size=7, padding=3)
        for _ in range(n_blocks):
            out_ch = min(512, ch * 2)
            blocks.append(_ResBlock1D(ch, out_ch, cond_dim, down=True, negative_slope=negative_slope))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(ch, 1)
        pos = torch.linspace(-1.0, 1.0, steps=self.n_snps).view(1, 1, -1)
        self.register_buffer("position", pos, persistent=False)
        self.negative_slope = negative_slope

    def forward(self, x, c):
        # x already has shape (packed_batch, pack_size, n_snps).
        if self.position_channel:
            pos = self.position.expand(x.shape[0], -1, -1)
            x = torch.cat([x, pos], dim=1)
        h = F.leaky_relu(self.stem(x), self.negative_slope)
        for block in self.blocks:
            h = block(h, c)
        h = h.mean(dim=-1)
        return self.out(h)


class GenotypeConvWGAN(_BaseWGANGP):
    """Residual 1-D cWGAN-GP that models contiguous SNP dosages directly.

    Input genotype dosages 0/1/2 are mapped to -1/0/+1 for training.  Generator
    output is continuous in [-1,1]. ``sample_dosage`` converts it to expected dosage
    [0,2], while ``sample_genotypes`` performs Binomial(2,p) sampling.
    """

    model_kind = "genotype_conv1d"

    def __init__(
        self,
        n_snps: int,
        latent_dim=64,
        cond_dim=3,
        channels=128,
        n_blocks=4,
        n_epochs=3000,
        batch_size=16,
        lr=1e-4,
        n_critic=5,
        lambda_gp=10.0,
        adam_beta1=0.0,
        adam_beta2=0.9,
        adam_eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        leaky_relu_slope=0.2,
        val_fraction=0.2,
        device="auto",
        seed=42,
        log_every=100,
        pack_size=2,
        balanced_sampling=False,
        position_channel=True,
        position_bias=True,
    ):
        super().__init__(TrainerConfig(
            latent_dim=latent_dim, cond_dim=cond_dim, n_epochs=n_epochs,
            batch_size=batch_size, lr=lr, n_critic=n_critic, lambda_gp=lambda_gp,
            adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
            weight_decay=weight_decay, amsgrad=amsgrad, val_fraction=val_fraction,
            device=device, seed=seed, log_every=log_every, pack_size=pack_size,
            balanced_sampling=balanced_sampling,
        ))
        self.n_snps = int(n_snps)
        self.channels = int(channels)
        self.n_blocks = int(n_blocks)
        self.leaky_relu_slope = leaky_relu_slope
        self.position_channel = bool(position_channel)
        self.position_bias = bool(position_bias)

    def _prepare_data(self, x):
        if x.shape[1] != self.n_snps:
            raise ValueError(f"Expected {self.n_snps} SNPs; received {x.shape[1]}.")
        if np.nanmin(x) < 0 or np.nanmax(x) > 2:
            raise ValueError("GenotypeConvWGAN expects imputed dosage values in [0,2].")
        return (x - 1.0).astype(np.float32)

    def _restore_samples(self, generated):
        return np.clip(generated + 1.0, 0.0, 2.0)

    def _build_networks(self, input_dim):
        g = _ConvGenerator(
            self.n_snps, self.config.latent_dim, self.config.cond_dim,
            self.channels, self.n_blocks, self.leaky_relu_slope, self.position_bias,
        )
        c = _ConvCritic(
            self.n_snps, self.config.cond_dim * self.config.pack_size,
            self.config.pack_size, max(16, self.channels // 4), self.n_blocks,
            self.leaky_relu_slope, self.position_channel,
        )
        return g, c

    def _pack(self, x, c):
        return _pack_sequence(x, c, self.config.pack_size)

    def sample_dosage(self, n_samples, conditions, z=None):
        return self.sample(n_samples, conditions, z=z)

    def sample_genotypes(self, n_samples, conditions, seed=None, z=None):
        dosage = self.sample_dosage(n_samples, conditions, z=z)
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        p = np.clip(dosage / 2.0, 0.0, 1.0)
        return rng.binomial(2, p).astype(np.int8)

    def _extra_checkpoint_state(self):
        return {
            "n_snps": self.n_snps,
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "leaky_relu_slope": self.leaky_relu_slope,
            "position_channel": self.position_channel,
            "position_bias": self.position_bias,
        }
