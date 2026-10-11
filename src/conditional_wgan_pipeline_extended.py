"""
Conditional WGAN-GP  —  PC-Space Synthetic Genotype Generator
==============================================================
Generates synthetic genotypes for a target Colombian state given its known
ancestry proportions (AFR / EUR / AMR).

Key differences from pca_synthetic_pipeline.py:
  - Trains on BAQ (admixed) + reference (pure AFR/EUR/AMR) samples jointly
  - Generator and critic are conditioned on a 3-dim admixture vector
  - At inference, ancestry proportions are sampled from a Dirichlet
    parameterised to match the target state, OR passed explicitly via CLI
  - The ML model drives generation end-to-end — no post-hoc Dirichlet
    replacement of the WGAN output

Pipeline:
  Load BAQ + Reference CSVs
  → Fit PCA on reference
  → Project both datasets into PC space
  → Assign admixture conditions
      BAQ      : barycentric projection onto reference centroid triangle
      Reference: known pure labels [1,0,0] / [0,1,0] / [0,0,1]
  → Train Conditional WGAN-GP
  → Sample with target admixture proportions
  → Inverse PCA → discretize → save genotypes + PCA plot

Usage:
    # Separate models in one timestamped run:
    python conditional_wgan_pipeline.py \
        --datasets BOG BAQ \
        --dataset_input BOG=data/BOG/genotypes.csv \
        --dataset_input BAQ=data/BAQ/genotypes.csv \
        --ref_input BOG=data/reference/reference_BOG.csv \
        --ref_input BAQ=data/reference/reference_BAQ.csv \
        --fam data/reference/reference.fam

    # Combined BOG+BAQ model plus a separate COR model. Quote braces in a shell.
    # BOG and BAQ must be harmonized upstream before using the acknowledgement flag.
    python conditional_wgan_pipeline.py \
        --datasets '{BOG,BAQ}' COR \
        --dataset_input BOG=data/BOG/genotypes.csv \
        --dataset_input BAQ=data/BAQ/genotypes.csv \
        --dataset_input COR=data/COR/genotypes.csv \
        --bog_baq_harmonized \
        --ref_input BOG=data/reference/reference_BOG.csv \
        --ref_input BAQ=data/reference/reference_BAQ.csv \
        --ref_input COR=data/reference/reference_COR.csv \
        --fam data/reference/reference.fam

    # Nested sample-size sweep: four independent BOG models and two COR models.
    # Smaller subsets are prefixes of one deterministic random permutation, so
    # BOG_n10 is a strict subset of BOG_n20, which is a strict subset of BOG_n30.
    python conditional_wgan_pipeline.py \
        --datasets BOG COR \
        --dataset_input BOG=data/BOG/genotypes.csv \
        --dataset_input COR=data/COR/genotypes.csv \
        --subset_samples "{'BOG': [10,20,30,'all'], 'COR': [10,20]}" \
        --ref_input BOG=data/reference/reference_BOG.csv \
        --ref_input COR=data/reference/reference_COR.csv \
        --fam data/reference/reference.fam

    # Legacy single-dataset invocation remains supported via --data_input.
"""

import argparse
import ast
import copy
import hashlib
import json
import itertools
import shutil
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from plot_colors import SYNTHETIC_COLOR, get_real_color
from cwgan_extensions.conditioning import load_external_conditions
from cwgan_extensions.models import FiLMConditionalWGAN, GenotypeConvWGAN


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# =============================================================================
# Population helpers  (shared with pca_synthetic_pipeline.py)
# =============================================================================

_POP_TO_SUPERPOP = {
    'ESN': 'AFR', 'GWD': 'AFR', 'YRI': 'AFR',
    'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR',
    'Kar': 'AMR', 'Karitiana': 'AMR',
    'May': 'AMR', 'Maya':      'AMR',
    'PEL': 'AMR',
    'Pim': 'AMR', 'Pima':  'AMR',
    'Sur': 'AMR', 'Surui': 'AMR',
}
_SUPERPOPS = ['AFR', 'EUR', 'AMR']

_POP_COLORS = {
    'AFR': '#2166AC',
    'EUR': '#E87D00',
    'AMR': '#89CFF0',
}


def load_population_labels(fam_path: str) -> list:
    labels = []
    with open(fam_path) as fh:
        for line in fh:
            cols = line.strip().split()
            if not cols:
                continue
            iid = cols[1]
            prefix = 'forReference'
            pop_code = (iid[len(prefix):].split('_')[0]
                        if iid.startswith(prefix) else iid.split('_')[0])
            labels.append(_POP_TO_SUPERPOP.get(pop_code, 'OTHER'))
    return labels


def _simplex_project_batch(pc_scores: np.ndarray,
                            vertices: np.ndarray) -> np.ndarray:
    """Least-squares barycentric projection onto the 3-vertex simplex."""
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    # Rewrite the triangle as two edge vectors relative to vertex v2 (AMR pole).
    # We solve: pc_score = α*v0 + β*v1 + γ*v2  subject to α+β+γ=1
    # → rearranged to: (pc_score - v2) = α*(v0-v2) + β*(v1-v2)
    A   = np.column_stack([v0 - v2, v1 - v2])   # (n_components, 2) basis
    X   = pc_scores - v2                          # (n_samples, n_components)
    ATA = A.T @ A                                 # (2, 2) normal equations
    XA  = X @ A                                   # (n_samples, 2) projections
    # Solve for [α, β] — these are the AFR and EUR proportions
    alpha01 = np.linalg.solve(ATA, XA.T).T
    # AMR proportion is determined by the simplex constraint α+β+γ=1
    alpha2  = 1.0 - alpha01[:, 0] - alpha01[:, 1]
    return np.column_stack([alpha01, alpha2[:, np.newaxis]])


def _ref_centroids(ref_pcs: np.ndarray, pop_labels: list) -> np.ndarray:
    """AFR / EUR / AMR reference population centroids in PC space (3 × n_components).

    These are the three poles of the ancestry triangle.  BAQ samples will be
    projected onto this triangle to estimate their AFR/EUR/AMR proportions.
    """
    labels   = np.array(pop_labels)
    vertices = []
    for sp in _SUPERPOPS:
        mask     = labels == sp
        # Average PC position of all samples from this ancestry group
        centroid = ref_pcs[mask].mean(axis=0)
        vertices.append(centroid)
        print(f"  {sp} centroid ({mask.sum():>3d} samples): "
              f"PC1={centroid[0]:+.2f}  PC2={centroid[1]:+.2f}", flush=True)
    return np.array(vertices, dtype=np.float32)


def estimate_baq_admixture(baq_pcs: np.ndarray,
                            ref_centroids: np.ndarray) -> np.ndarray:
    """Estimate BAQ admixture proportions via barycentric projection.

    Returns:
        (n_baq, 3) array of non-negative admixture proportions summing to 1.
    """
    bary = _simplex_project_batch(baq_pcs, ref_centroids)
    # Least-squares can produce negative weights for samples that fall outside
    # the triangle — clip to zero and renormalise to keep on the simplex.
    bary = np.clip(bary, 0.0, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return bary.astype(np.float32)


# =============================================================================
# Conditional WGAN-GP
# =============================================================================

class _CondNetG(nn.Module):
    """Conditional generator: concat(z, c) -> PC scores."""

    def __init__(self, latent_dim: int, cond_dim: int,
                 hidden_dims: list, output_dim: int,
                 negative_slope: float = 0.2,
                 layernorm_eps: float = 1e-5):
        super().__init__()
        layers, prev = [], latent_dim + cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h, eps=layernorm_eps),
                       nn.LeakyReLU(negative_slope, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=1))


class _CondNetC(nn.Module):
    """Conditional critic: concat(x, c) -> unbounded Wasserstein score."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dims: list,
                 negative_slope: float = 0.2):
        super().__init__()
        layers, prev = [], input_dim + cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(negative_slope, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=1))


class ConditionalWGAN:
    """Conditional WGAN-GP with explicit hyperparameters and loss history."""

    def __init__(self,
                 latent_dim: int = 32,
                 cond_dim: int = 3,
                 hidden_dims: list = None,
                 n_epochs: int = 3000,
                 batch_size: int = 64,
                 lr: float = 1e-4,
                 n_critic: int = 5,
                 lambda_gp: float = 10.0,
                 adam_beta1: float = 0.0,
                 adam_beta2: float = 0.9,
                 adam_eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 leaky_relu_slope: float = 0.2,
                 layernorm_eps: float = 1e-5,
                 norm_eps: float = 1e-8,
                 val_fraction: float = 0.2,
                 device: str = None,
                 seed: int = 42,
                 log_every: int = 500):
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.leaky_relu_slope = leaky_relu_slope
        self.layernorm_eps = layernorm_eps
        self.norm_eps = norm_eps
        self.val_fraction = val_fraction
        self.device_str = device
        self.seed = seed
        self.log_every = log_every
        self._G = self._C = self._device = None
        self._mean = self._std = None
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
        if self.device_str and self.device_str.lower() != 'auto':
            return torch.device(self.device_str)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                          cond: torch.Tensor) -> torch.Tensor:
        n = real.size(0)
        alpha = torch.rand(n, 1, device=self._device).expand_as(real)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        grads = torch.autograd.grad(
            outputs=self._C(interp, cond), inputs=interp,
            grad_outputs=torch.ones(n, 1, device=self._device),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    @staticmethod
    def _param_count(net: nn.Module) -> int:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def _validation_losses(self, loader, G, C_net):
        if loader is None:
            return dict(g=np.nan, c_total=np.nan, c_wasserstein=np.nan, gp=np.nan)

        G.eval(); C_net.eval()
        sums = dict(g=0.0, c_total=0.0, c_wasserstein=0.0, gp=0.0)
        n_batches = 0
        for real_batch, cond_batch in loader:
            real = real_batch.to(self._device)
            cond = cond_batch.to(self._device)
            b = real.size(0)
            with torch.no_grad():
                z = torch.randn(b, self.latent_dim, device=self._device)
                fake = G(z, cond)
                c_fake = C_net(fake, cond).mean()
                c_real = C_net(real, cond).mean()
                g_loss = -c_fake
                wasserstein = c_fake - c_real
            gp = self._gradient_penalty(real, fake.detach(), cond)
            c_total = wasserstein + self.lambda_gp * gp
            sums['g'] += float(g_loss.item())
            sums['c_total'] += float(c_total.item())
            sums['c_wasserstein'] += float(wasserstein.item())
            sums['gp'] += float(gp.item())
            n_batches += 1
        G.train(); C_net.train()
        return {k: v / max(n_batches, 1) for k, v in sums.items()}

    def fit(self, pc_scores: np.ndarray, conditions: np.ndarray, epoch_callback=None):
        """Train and return a per-epoch train/validation loss history."""
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError('val_fraction must be in [0, 1).')
        if self.cond_dim != conditions.shape[1]:
            raise ValueError(
                f'cond_dim={self.cond_dim}, but conditions have {conditions.shape[1]} columns.'
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_samples, n_dim = pc_scores.shape
        self._device = self._resolve_device()

        self._mean = pc_scores.mean(axis=0).astype(np.float32)
        self._std = (pc_scores.std(axis=0) + self.norm_eps).astype(np.float32)
        normed = ((pc_scores - self._mean) / self._std).astype(np.float32)

        # Fixed split for reproducible model selection/diagnostics.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_samples)
        n_val = int(round(n_samples * self.val_fraction))
        if self.val_fraction > 0 and n_samples >= 5:
            n_val = max(1, min(n_val, n_samples - 2))
        else:
            n_val = 0
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        self.val_indices = val_idx
        self.train_indices = train_idx

        X_train = torch.from_numpy(normed[train_idx])
        C_train = torch.from_numpy(conditions[train_idx].astype(np.float32))
        bs = min(self.batch_size, len(train_idx))
        train_loader = DataLoader(
            TensorDataset(X_train, C_train), batch_size=bs, shuffle=True,
            drop_last=(len(train_idx) > bs),
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = None
        if n_val:
            X_val = torch.from_numpy(normed[val_idx])
            C_val = torch.from_numpy(conditions[val_idx].astype(np.float32))
            val_loader = DataLoader(
                TensorDataset(X_val, C_val),
                batch_size=min(self.batch_size, n_val), shuffle=False,
            )

        print(f"  cWGAN device    : {self._device}", flush=True)
        print(f"  Training data   : {n_samples} samples x {n_dim} PC dims", flush=True)
        print(f"  Split           : train={len(train_idx)}  val={n_val} "
              f"(val_fraction={self.val_fraction:.3f})", flush=True)
        print(f"  Cond dim        : {self.cond_dim}", flush=True)
        print(f"  latent={self.latent_dim}  hidden={self.hidden_dims}  "
              f"epochs={self.n_epochs}  bs={self.batch_size}  lr={self.lr:g}  "
              f"n_critic={self.n_critic}  lambda_gp={self.lambda_gp:g}", flush=True)

        G = _CondNetG(
            self.latent_dim, self.cond_dim, self.hidden_dims, n_dim,
            negative_slope=self.leaky_relu_slope,
            layernorm_eps=self.layernorm_eps,
        ).to(self._device)
        C_net = _CondNetC(
            n_dim, self.cond_dim, self.hidden_dims,
            negative_slope=self.leaky_relu_slope,
        ).to(self._device)
        opt_G = torch.optim.Adam(
            G.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
        )
        opt_C = torch.optim.Adam(
            C_net.parameters(), lr=self.lr, betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
        )
        G.train(); C_net.train()
        self._G = G
        self._C = C_net
        # Capture the randomly initialized state before ANY optimizer update.
        # This is semantic checkpoint 'baseline' (epoch 0).
        self._baseline_state = {
            'G': copy.deepcopy(G.state_dict()),
            'C': copy.deepcopy(C_net.state_dict()),
        }

        print("\n  Generator architecture:\n" + str(G), flush=True)
        print(f"  Generator trainable parameters: {self._param_count(G):,}", flush=True)
        print("\n  Critic architecture:\n" + str(C_net), flush=True)
        print(f"  Critic trainable parameters: {self._param_count(C_net):,}", flush=True)
        print(f"\n  Optimizer G: {opt_G}", flush=True)
        print(f"  Optimizer C: {opt_C}", flush=True)
        print("  Loss G      : -E[C(G(z,c), c)]", flush=True)
        print("  Loss C      : E[C(fake,c)] - E[C(real,c)] + lambda_gp * GP", flush=True)
        print("  GP          : E[(||grad_xhat C(xhat,c)||_2 - 1)^2]", flush=True)

        history = {
            'epoch': [],
            'train_g': [], 'val_g': [],
            'train_c_total': [], 'val_c_total': [],
            'train_c_wasserstein': [], 'val_c_wasserstein': [],
            'train_gp': [], 'val_gp': [],
        }
        t_train = time.time()

        for epoch in range(1, self.n_epochs + 1):
            sums = dict(g=0.0, c_total=0.0, c_wasserstein=0.0, gp=0.0)
            n_g = n_c = 0
            for real_batch, cond_batch in train_loader:
                real = real_batch.to(self._device)
                cond = cond_batch.to(self._device)
                b = real.size(0)

                for _ in range(self.n_critic):
                    z = torch.randn(b, self.latent_dim, device=self._device)
                    fake = G(z, cond).detach()
                    gp = self._gradient_penalty(real, fake, cond)
                    c_fake = C_net(fake, cond).mean()
                    c_real = C_net(real, cond).mean()
                    wasserstein = c_fake - c_real
                    c_total = wasserstein + self.lambda_gp * gp
                    opt_C.zero_grad(set_to_none=True)
                    c_total.backward()
                    opt_C.step()
                    sums['c_total'] += float(c_total.item())
                    sums['c_wasserstein'] += float(wasserstein.item())
                    sums['gp'] += float(gp.item())
                    n_c += 1

                z = torch.randn(b, self.latent_dim, device=self._device)
                g_loss = -C_net(G(z, cond), cond).mean()
                opt_G.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_G.step()
                sums['g'] += float(g_loss.item())
                n_g += 1

            train_losses = {
                'g': sums['g'] / max(n_g, 1),
                'c_total': sums['c_total'] / max(n_c, 1),
                'c_wasserstein': sums['c_wasserstein'] / max(n_c, 1),
                'gp': sums['gp'] / max(n_c, 1),
            }
            val_losses = self._validation_losses(val_loader, G, C_net)

            history['epoch'].append(epoch)
            for key in ('g', 'c_total', 'c_wasserstein', 'gp'):
                history[f'train_{key}'].append(train_losses[key])
                history[f'val_{key}'].append(val_losses[key])

            callback_metrics = None
            if epoch_callback is not None:
                callback_metrics = epoch_callback(self, epoch)
                if callback_metrics and 'selection_score' in callback_metrics:
                    score = float(callback_metrics['selection_score'])
                    if np.isfinite(score) and (self._best_score is None or score < self._best_score):
                        self._best_score = score
                        self._best_epoch = epoch
                        self._best_state = {
                            'G': copy.deepcopy(G.state_dict()),
                            'C': copy.deepcopy(C_net.state_dict()),
                        }
                G.train(); C_net.train()

            if epoch == 1:
                self._first_state = {
                    'G': copy.deepcopy(G.state_dict()),
                    'C': copy.deepcopy(C_net.state_dict()),
                }

            if self.log_every > 0 and (epoch == 1 or epoch % self.log_every == 0 or epoch == self.n_epochs):
                msg = (
                    f"  [cWGAN epoch {epoch:>5}/{self.n_epochs}]  "
                    f"train G={train_losses['g']:+.4f} "
                    f"C={train_losses['c_total']:+.4f} "
                    f"W={train_losses['c_wasserstein']:+.4f} "
                    f"GP={train_losses['gp']:.4f}"
                )
                if val_loader is not None:
                    msg += (
                        f"  |  val G={val_losses['g']:+.4f} "
                        f"C={val_losses['c_total']:+.4f} "
                        f"W={val_losses['c_wasserstein']:+.4f} "
                        f"GP={val_losses['gp']:.4f}"
                    )
                if callback_metrics and 'selection_score' in callback_metrics:
                    msg += f"  |  selection={callback_metrics['selection_score']:.4f}"
                msg += f"  [{_elapsed(t_train)}]"
                print(msg, flush=True)

        self._last_state = {
            'G': copy.deepcopy(G.state_dict()),
            'C': copy.deepcopy(C_net.state_dict()),
        }
        if self._best_state is not None:
            G.load_state_dict(self._best_state['G'])
            C_net.load_state_dict(self._best_state['C'])
            print(
                f"  Restored best checkpoint-selection state from epoch {self._best_epoch} "
                f"(selection_score={self._best_score:.6g}).", flush=True
            )
        G.eval(); C_net.eval()
        self._G = G
        self.history = history
        print(f"  cWGAN training complete  [{_elapsed(t_train)}]", flush=True)
        return history

    def sample(self, n_samples: int, target_admixture: np.ndarray) -> np.ndarray:
        if self._G is None:
            raise RuntimeError("Call fit() before sample()")
        self._G.eval()
        cond = np.asarray(target_admixture, dtype=np.float32)
        if cond.ndim == 1:
            cond = np.tile(cond, (n_samples, 1))
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self._device)
            c_t = torch.from_numpy(cond).to(self._device)
            norm = self._G(z, c_t).cpu().numpy()
        return (norm * self._std + self._mean).astype(np.float32)

    def _state_dicts_for(self, state: str):
        if self._G is None or self._C is None:
            raise RuntimeError('Model must be fit before accessing checkpoint states.')
        if state == 'current':
            return {'G': self._G.state_dict(), 'C': self._C.state_dict()}
        if state == 'baseline':
            if self._baseline_state is None:
                raise RuntimeError('Baseline pre-training state is unavailable; fit/initialize the model first.')
            return self._baseline_state
        if state == 'first':
            if self._first_state is None:
                raise RuntimeError('First-epoch state is unavailable; fit the model first.')
            return self._first_state
        if state == 'last':
            if self._last_state is None:
                raise RuntimeError('Final-epoch state is unavailable; fit the model first.')
            return self._last_state
        if state == 'best':
            if self._best_state is None:
                raise RuntimeError('Best-validation state is unavailable; enable validation checkpointing.')
            return self._best_state
        raise ValueError("state must be one of: current, baseline, first, best, last")

    def activate_checkpoint_state(self, state: str):
        """Load one in-memory checkpoint state for sampling/evaluation."""
        state_dicts = self._state_dicts_for(state)
        self._G.load_state_dict(state_dicts['G'])
        self._C.load_state_dict(state_dicts['C'])
        self._G.eval(); self._C.eval()

    def save_checkpoint(self, path: str, state: str = 'current'):
        state_dicts = self._state_dicts_for(state)
        torch.save({
            'checkpoint_state': state,
            'generator_state_dict': state_dicts['G'],
            'critic_state_dict': state_dicts['C'],
            'pc_mean': self._mean,
            'pc_std': self._std,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'baseline_epoch': 0 if self._baseline_state is not None else None,
            'first_epoch': 1 if self._first_state is not None else None,
            'best_epoch': self._best_epoch,
            'best_score': self._best_score,
            'last_epoch': self.n_epochs if self._last_state is not None else None,
            'hyperparameters': {
                'latent_dim': self.latent_dim,
                'cond_dim': self.cond_dim,
                'hidden_dims': self.hidden_dims,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'n_critic': self.n_critic,
                'lambda_gp': self.lambda_gp,
                'adam_beta1': self.adam_beta1,
                'adam_beta2': self.adam_beta2,
                'adam_eps': self.adam_eps,
                'weight_decay': self.weight_decay,
                'amsgrad': self.amsgrad,
                'leaky_relu_slope': self.leaky_relu_slope,
                'layernorm_eps': self.layernorm_eps,
                'norm_eps': self.norm_eps,
                'val_fraction': self.val_fraction,
                'seed': self.seed,
            },
        }, path)
        print(f"  Model checkpoint saved ({state}): {path}", flush=True)


# =============================================================================
# PCA Compression
# =============================================================================

class PCACompressor:
    def __init__(self, n_components: int = None):
        self.n_components  = n_components
        self.pca           = None

    def fit_transform(self, genotypes: np.ndarray) -> np.ndarray:
        n_samples, n_snps = genotypes.shape
        if self.n_components is None:
            self.n_components = min(n_samples - 1, n_snps, 50)
        print(f"  Input shape : {n_samples} samples × {n_snps} SNPs")
        print(f"  Retaining   : {self.n_components} PCs")
        self.pca = PCA(n_components=self.n_components,
                       svd_solver='randomized', random_state=42)
        pc_scores = self.pca.fit_transform(genotypes)
        var_exp   = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"  Variance explained: {var_exp:.2f}%")
        return pc_scores

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        return self.pca.transform(genotypes)

    def inverse_transform(self, pc_scores: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(pc_scores)


def discretize_genotypes(continuous: np.ndarray, seed: int = 42) -> np.ndarray:
    # Binomial sampling instead of rounding.
    #
    # Why rounding is wrong:
    #   Lossy inverse PCA pushes many SNP values toward the mean, which for
    #   admixed samples is often near 1.0.  np.rint() then assigns every one
    #   of those SNPs to heterozygous (1), creating systematic het inflation
    #   (~37% excess hets) that dominates PC1 in a joint PLINK2 PCA and
    #   completely separates synthetic from real samples.
    #
    # Why Binomial works:
    #   Treat the continuous value v as an expected diploid dosage.  Allele
    #   probability p = v/2.  Sample two independent Bernoulli(p) draws —
    #   i.e. Binomial(n=2, p=v/2).  This preserves E[dosage] = v while
    #   producing realistic 0/1/2 variance.  At v=1 (p=0.5): 25% hom-ref,
    #   50% het, 25% hom-alt — matching Hardy-Weinberg expectations for a
    #   common variant, instead of 100% het from rounding.
    rng = np.random.default_rng(seed)
    p   = np.clip(continuous / 2.0, 0.0, 1.0)   # (n_samples, n_snps)
    return rng.binomial(2, p).astype(np.int8)


# =============================================================================
# Inference-time admixture sampling
# =============================================================================

def sample_target_admixture(n_samples: int,
                             target_admixture: np.ndarray,
                             alpha_concentration: float = 20.0,
                             seed: int = 42) -> np.ndarray:
    """Sample per-sample admixture proportions around a target mean.

    Uses a Dirichlet distribution centred on target_admixture.
    alpha_concentration controls how tightly samples cluster around the target:
        high (≥20): samples very close to the target mean  (conservative)
        low  (~2):  samples spread across the triangle      (exploratory)

    Args:
        target_admixture:    (3,) mean admixture [AFR, EUR, AMR], sums to 1.
        alpha_concentration: Dirichlet precision parameter.
    """
    rng   = np.random.default_rng(seed)
    alpha = target_admixture * alpha_concentration
    alpha = np.maximum(alpha, 0.01)
    props = rng.dirichlet(alpha, size=n_samples).astype(np.float32)
    return props


# =============================================================================
# PCA Overlay Plot
# =============================================================================

def plot_pca_overlay(baq_pcs: np.ndarray,
                     synth_pcs: np.ndarray,
                     var_explained: np.ndarray,
                     save_path: str,
                     ref_pcs: np.ndarray = None,
                     ref_pop_labels: list = None,
                     triangle_vertices: np.ndarray = None,
                     baq_vertex_indices: dict = None,
                     target_admixture: np.ndarray = None,
                     job_id: str = "",
                     seed: int = 42,
                     sample_name: str = "BAQ",
                     real_color: str = None):

    real_color = real_color or get_real_color(sample_name)
    var_pct    = var_explained[:2] * 100
    ref_labels = np.array(ref_pop_labels) if ref_pop_labels is not None else None

    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference pure samples
    if ref_pcs is not None and ref_labels is not None:
        for sp in _SUPERPOPS:
            mask = ref_labels == sp
            if mask.any():
                ax.scatter(ref_pcs[mask, 0], ref_pcs[mask, 1],
                           c=_POP_COLORS[sp], alpha=0.40, s=22,
                           label=sp, edgecolors='none', zorder=1)

    # Ancestry triangle
    if triangle_vertices is not None:
        pts = np.vstack([triangle_vertices[:, :2], triangle_vertices[0, :2]])
        ax.plot(pts[:, 0], pts[:, 1], color='#333333',
                linestyle='--', linewidth=1.2, alpha=0.6,
                zorder=4, label='Ancestry triangle')
        for i, sp in enumerate(_SUPERPOPS):
            ax.scatter(triangle_vertices[i, 0], triangle_vertices[i, 1],
                       marker='^', s=90, color=_POP_COLORS[sp],
                       edgecolors='black', linewidths=0.7, zorder=6)

    # Real samples
    ax.scatter(baq_pcs[:, 0], baq_pcs[:, 1],
               c=real_color, alpha=0.9, s=50, label=f'Real ({sample_name})',
               edgecolors='white', linewidths=0.4, zorder=2)

    # Extreme sample star markers
    if baq_vertex_indices is not None:
        for sp, idx in baq_vertex_indices.items():
            ax.plot(baq_pcs[idx, 0], baq_pcs[idx, 1],
                    marker='*', markersize=16, color=real_color,
                    markeredgecolor='white', markeredgewidth=0.5,
                    linestyle='none', zorder=7)

    # Synthetic samples
    ax.scatter(synth_pcs[:, 0], synth_pcs[:, 1],
               c=SYNTHETIC_COLOR, alpha=0.7, s=50, label='Synthetic (cWGAN)',
               edgecolors='white', linewidths=0.4, zorder=3)

    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)", fontsize=12)
    ax.legend(fontsize=9, markerscale=0.9, loc='best')

    title_parts = ["cWGAN | PC1 vs PC2: Real · Synthetic · Reference"]
    if target_admixture is not None:
        afr, eur, amr = target_admixture
        title_parts.append(f"target AFR={afr:.2f} EUR={eur:.2f} AMR={amr:.2f}")
    title_parts.append(f"seed={seed}")
    if job_id:
        title_parts.append(f"job {job_id}")
    ax.set_title("  |  ".join(title_parts), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}", flush=True)


# =============================================================================
# BAQ extreme-vertex finder  (for plot star markers)
# =============================================================================

def _find_baq_vertices(baq_pcs: np.ndarray,
                        ref_pcs: np.ndarray,
                        pop_labels: list) -> tuple:
    labels = np.array(pop_labels)
    vertex_pcs     = []
    vertex_indices = {}
    for sp in _SUPERPOPS:
        mask     = labels == sp
        centroid = ref_pcs[mask].mean(axis=0)
        dists    = np.linalg.norm(baq_pcs - centroid, axis=1)
        idx      = int(np.argmin(dists))
        vertex_pcs.append(baq_pcs[idx])
        vertex_indices[sp] = idx
    return np.array(vertex_pcs), vertex_indices


# =============================================================================
# Training diagnostics and synthetic-genotype quality metrics
# =============================================================================

def _group_checkpoint_labels_by_epoch(checkpoint_epochs: dict):
    grouped = {}
    for label, epoch in (checkpoint_epochs or {}).items():
        if epoch is None:
            continue
        grouped.setdefault(int(epoch), []).append(str(label))
    return {epoch: sorted(labels) for epoch, labels in sorted(grouped.items())}

def _draw_checkpoint_lines(ax, checkpoint_epochs: dict):
    """Draw one vertical line per unique saved epoch, merging overlapping labels."""
    for epoch, labels in _group_checkpoint_labels_by_epoch(checkpoint_epochs).items():
        joined = '/'.join(labels)
        ax.axvline(epoch, linestyle='--', linewidth=1.1, alpha=0.8,
                   label=f'saved {joined} (epoch {epoch})')

def plot_loss_history(history: dict, save_path: str, checkpoint_epochs: dict = None):
    """Plot train/validation losses and mark epochs whose weights are saved."""
    epochs = np.asarray(history['epoch'])
    panels = [
        ('g', 'Generator loss'),
        ('c_total', 'Critic total loss'),
        ('c_wasserstein', 'Critic Wasserstein component'),
        ('gp', 'Gradient penalty'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(epochs, history[f'train_{key}'], label='train', linewidth=1.3)
        vals = np.asarray(history[f'val_{key}'], dtype=float)
        if np.isfinite(vals).any():
            ax.plot(epochs, vals, label='validation', linewidth=1.3)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.2)
        _draw_checkpoint_lines(ax, checkpoint_epochs)
        ax.legend()
    fig.suptitle('Conditional WGAN-GP training diagnostics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Loss plot saved: {save_path}", flush=True)



def _validation_selection_components(metrics: dict) -> dict:
    """Return bounded normalized fidelity errors used for checkpoint selection.

    Each component is scaled to [0,1] using its natural range:
      MAF MAE / 0.5; heterozygosity difference / 1; LD r^2 MAE / 1;
      RBF MMD^2 / 2 (for kernels in [0,1]); and |NNAA-0.5| / 0.5.
    The composite selection score is the mean of finite components. Lower is better.
    """
    raw = {
        'maf_mae': float(metrics.get('maf_mae', np.nan)),
        'heterozygosity_abs_diff': float(metrics.get('heterozygosity_mean_abs_diff', np.nan)),
        'ld_r2_mae': float(metrics.get('ld_adjacent_r2_mae', np.nan)),
        'pc_mmd2': float(metrics.get('pc_rbf_mmd2', np.nan)),
        'nnaa_abs_deviation': abs(float(metrics.get('pc_adversarial_accuracy_ts', np.nan)) - 0.5),
    }
    candidate_normalized = {
        'norm_maf_mae': raw['maf_mae'] / 0.5,
        'norm_heterozygosity_abs_diff': raw['heterozygosity_abs_diff'],
        'norm_ld_r2_mae': raw['ld_r2_mae'],
        'norm_pc_mmd2': raw['pc_mmd2'] / 2.0,
        'norm_nnaa_abs_deviation': raw['nnaa_abs_deviation'] / 0.5,
    }
    n_finite = sum(np.isfinite(v) for v in candidate_normalized.values())
    # Keep the composite comparable across epochs: undefined components receive the
    # worst bounded error (1.0) rather than disappearing from the denominator.
    normalized = {
        k: float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 1.0
        for k, v in candidate_normalized.items()
    }
    score = float(np.mean(list(normalized.values())))
    return {**raw, **normalized, 'composite_selection_score': score,
            'n_finite_components': int(n_finite)}


def plot_validation_selection_history(records: list, save_path: str, checkpoint_epochs: dict = None):
    """Plot checkpoint-selection score/components and mark saved model epochs.

    With val_fraction > 0 this is a held-out validation score.  With
    val_fraction == 0 it is deliberately an in-sample score against the query
    samples used for fitting, and the plot is labelled accordingly.
    """
    if not records:
        return
    df = pd.DataFrame(records)
    if 'selection_score' not in df or not np.isfinite(df['selection_score']).any():
        return
    best_i = int(np.nanargmin(df['selection_score'].to_numpy(dtype=float)))
    best_epoch = int(df.iloc[best_i]['epoch'])
    best_score = float(df.iloc[best_i]['selection_score'])
    bases = set(df['evaluation_basis'].dropna().astype(str)) if 'evaluation_basis' in df else set()
    in_sample_only = bases == {'training_in_sample'}
    score_label = 'in-sample training selection score' if in_sample_only else 'validation selection score'
    title = (
        'In-sample training synthetic-data fidelity used for checkpoint selection'
        if in_sample_only else
        'Validation synthetic-data fidelity used for checkpoint selection'
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df['epoch'], df['selection_score'], marker='o', markersize=3,
                 linewidth=1.3, label=score_label)
    axes[0].scatter([best_epoch], [best_score], s=45, zorder=4,
                    label=f'minimum score (epoch {best_epoch})')
    _draw_checkpoint_lines(axes[0], checkpoint_epochs)
    axes[0].set_ylabel('Composite score (lower is better)')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    component_cols = [
        ('norm_maf_mae', 'MAF MAE'),
        ('norm_heterozygosity_abs_diff', 'heterozygosity'),
        ('norm_ld_r2_mae', 'LD $r^2$ MAE'),
        ('norm_pc_mmd2', 'PC MMD$^2$'),
        ('norm_nnaa_abs_deviation', 'NNAA deviation'),
    ]
    for col, label in component_cols:
        if col in df and np.isfinite(df[col].to_numpy(dtype=float)).any():
            axes[1].plot(df['epoch'], df[col], linewidth=1.1, label=label)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Normalized component error')
    axes[1].grid(alpha=0.2)
    _draw_checkpoint_lines(axes[1], checkpoint_epochs)
    axes[1].legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Checkpoint-selection plot saved: {save_path}", flush=True)


def _snapshot_rng_state():
    state = {
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': None,
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _make_validation_checkpoint_callback(model_backend: str,
                                         all_geno: np.ndarray,
                                         all_admix: np.ndarray,
                                         compressor,
                                         n_query_samples: int,
                                         n_epochs: int,
                                         seed: int,
                                         eval_every: int,
                                         max_samples: int,
                                         max_ld_pairs: int,
                                         selection_metric: str,
                                         records: list):
    """Build an epoch callback that scores generated data against real samples.

    If a validation split exists, the real comparison cohort is the held-out
    query population (falling back to all held-out rows when necessary).  If
    val_fraction == 0, the analogous query-population training rows are used
    instead.  The latter is an in-sample model-selection score, not an estimate
    of out-of-sample generalization.
    """
    warned = {'no_comparison': False, 'too_small': False}

    def callback(model, epoch: int):
        if not (epoch == 1 or epoch % eval_every == 0 or epoch == n_epochs):
            return None

        val_idx = getattr(model, 'val_indices', None)
        val_idx = np.asarray(val_idx, dtype=int) if val_idx is not None else np.array([], dtype=int)
        if len(val_idx) > 0:
            # Prefer query-population holdout samples because those match the intended generation target.
            query_val = val_idx[val_idx < n_query_samples]
            evaluation_basis = 'held_out_validation'
            scope = 'query_holdout'
            candidate = query_val
            if len(candidate) < 2:
                candidate = val_idx
                scope = 'all_holdout_fallback'
        else:
            train_idx = getattr(model, 'train_indices', None)
            train_idx = np.asarray(train_idx, dtype=int) if train_idx is not None else np.array([], dtype=int)
            if len(train_idx) == 0:
                if not warned['no_comparison']:
                    print("  [checkpoint selection] No validation or training indices available; selection disabled.", flush=True)
                    warned['no_comparison'] = True
                return None
            # val_fraction=0: score synthetic samples against the query rows that were used to train.
            query_train = train_idx[train_idx < n_query_samples]
            evaluation_basis = 'training_in_sample'
            scope = 'query_train_in_sample'
            candidate = query_train
            if len(candidate) < 2:
                candidate = train_idx
                scope = 'all_train_in_sample_fallback'

        if len(candidate) < 2:
            if not warned['too_small']:
                print(
                    "  [checkpoint selection] Fewer than 2 real comparison samples; selection disabled.",
                    flush=True,
                )
                warned['too_small'] = True
            return None

        if max_samples > 0 and len(candidate) > max_samples:
            choose_rng = np.random.default_rng(seed + 104729)
            candidate = np.sort(choose_rng.choice(candidate, size=max_samples, replace=False))

        real_geno = all_geno[candidate]
        cond = all_admix[candidate]
        n_eval = len(candidate)

        # Use identical random latent draws/discretization at every scored epoch so changes in
        # the metric are driven by model weights rather than Monte-Carlo noise.
        rng_state = _snapshot_rng_state()
        try:
            eval_seed = seed + 65537
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)
            if model_backend == 'genotype_conv':
                synth_geno = model.sample_genotypes(n_eval, cond, seed=eval_seed)
            else:
                synth_pc_cont = model.sample(n_eval, cond)
                synth_geno = discretize_genotypes(
                    compressor.inverse_transform(synth_pc_cont), seed=eval_seed
                )
            real_pcs = compressor.transform(real_geno.astype(np.float32))
            synth_pcs = compressor.transform(synth_geno.astype(np.float32))
            metrics, _ = compute_quality_metrics(
                real_geno, synth_geno, real_pcs, synth_pcs,
                max_ld_pairs=max_ld_pairs,
            )
        finally:
            _restore_rng_state(rng_state)

        comps = _validation_selection_components(metrics)
        if selection_metric == 'composite':
            score = comps['composite_selection_score']
        elif selection_metric == 'pc_mmd2':
            score = comps['pc_mmd2']
        else:
            raise ValueError(f"Unknown checkpoint selection metric: {selection_metric}")

        record = {
            'epoch': int(epoch),
            'evaluation_basis': evaluation_basis,
            'scope': scope,
            'n_comparison_samples': int(n_eval),
            # Retained for backwards compatibility with prior analysis scripts.
            'n_validation_samples': int(n_eval) if evaluation_basis == 'held_out_validation' else 0,
            'n_training_comparison_samples': int(n_eval) if evaluation_basis == 'training_in_sample' else 0,
            'selection_metric': selection_metric,
            'selection_score': float(score),
            **comps,
        }
        records.append(record)
        print(
            f"  [checkpoint selection epoch {epoch}] score={score:.6g} "
            f"MAF={comps['maf_mae']:.4g} HET={comps['heterozygosity_abs_diff']:.4g} "
            f"LD={comps['ld_r2_mae']:.4g} MMD2={comps['pc_mmd2']:.4g} "
            f"NNAAdev={comps['nnaa_abs_deviation']:.4g} n={n_eval} "
            f"({scope}; {evaluation_basis})",
            flush=True,
        )
        return {'selection_score': float(score)}

    return callback

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float('nan')
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def _rbf_mmd2(x: np.ndarray, y: np.ndarray) -> tuple:
    """Biased non-negative RBF MMD^2 with pooled median-distance bandwidth."""
    pooled = np.vstack([x, y]).astype(np.float64)
    d2 = _pairwise_sq_dists(pooled, pooled)
    upper = d2[np.triu_indices_from(d2, k=1)]
    positive = upper[upper > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, 1e-12)

    kxx = np.exp(-_pairwise_sq_dists(x, x) / (2.0 * sigma2))
    kyy = np.exp(-_pairwise_sq_dists(y, y) / (2.0 * sigma2))
    kxy = np.exp(-_pairwise_sq_dists(x, y) / (2.0 * sigma2))
    mmd2 = float(max(kxx.mean() + kyy.mean() - 2.0 * kxy.mean(), 0.0))
    return mmd2, float(np.sqrt(sigma2))


def _adversarial_accuracy(real_pc: np.ndarray, synth_pc: np.ndarray) -> dict:
    """Nearest-neighbour adversarial accuracy; ideal mixing is near 0.5."""
    rr = np.sqrt(_pairwise_sq_dists(real_pc, real_pc))
    ss = np.sqrt(_pairwise_sq_dists(synth_pc, synth_pc))
    rs = np.sqrt(_pairwise_sq_dists(real_pc, synth_pc))
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(ss, np.inf)
    d_rr = rr.min(axis=1)
    d_rs = rs.min(axis=1)
    d_ss = ss.min(axis=1)
    d_sr = rs.min(axis=0)
    aa_truth = float(np.mean(d_rr < d_rs))
    aa_syn = float(np.mean(d_ss < d_sr))
    return {
        'aa_truth': aa_truth,
        'aa_synthetic': aa_syn,
        'aa_ts': 0.5 * (aa_truth + aa_syn),
    }


def _adjacent_ld_r2(geno: np.ndarray, pair_starts: np.ndarray) -> np.ndarray:
    out = np.full(len(pair_starts), np.nan, dtype=np.float64)
    for j, idx in enumerate(pair_starts):
        a = geno[:, idx].astype(np.float64)
        b = geno[:, idx + 1].astype(np.float64)
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            r = np.corrcoef(a, b)[0, 1]
            out[j] = r * r
    return out


def compute_quality_metrics(real_geno: np.ndarray,
                            synth_geno: np.ndarray,
                            real_pcs: np.ndarray,
                            synth_pcs: np.ndarray,
                            max_ld_pairs: int = 2000) -> tuple:
    """Compute fidelity/coverage metrics for synthetic genotype data."""
    real = np.asarray(real_geno, dtype=np.float64)
    synth = np.asarray(synth_geno, dtype=np.float64)

    af_real = real.mean(axis=0) / 2.0
    af_synth = synth.mean(axis=0) / 2.0
    maf_real = np.minimum(af_real, 1.0 - af_real)
    maf_synth = np.minimum(af_synth, 1.0 - af_synth)

    het_real = (real == 1).mean(axis=1)
    het_synth = (synth == 1).mean(axis=1)

    n_pairs = min(max_ld_pairs, max(real.shape[1] - 1, 0))
    if n_pairs > 0:
        pair_starts = np.unique(np.linspace(0, real.shape[1] - 2, n_pairs, dtype=int))
        ld_real = _adjacent_ld_r2(real, pair_starts)
        ld_synth = _adjacent_ld_r2(synth, pair_starts)
        ld_mask = np.isfinite(ld_real) & np.isfinite(ld_synth)
    else:
        pair_starts = np.array([], dtype=int)
        ld_real = ld_synth = np.array([], dtype=float)
        ld_mask = np.array([], dtype=bool)

    d = min(real_pcs.shape[1], synth_pcs.shape[1], 10)
    real_pc = real_pcs[:, :d].astype(np.float64)
    synth_pc = synth_pcs[:, :d].astype(np.float64)
    center = real_pc.mean(axis=0)
    scale = real_pc.std(axis=0) + 1e-8
    real_pc_z = (real_pc - center) / scale
    synth_pc_z = (synth_pc - center) / scale
    mmd2, mmd_sigma = _rbf_mmd2(real_pc_z, synth_pc_z)
    aa = _adversarial_accuracy(real_pc_z, synth_pc_z)

    metrics = {
        'n_real_samples': int(real.shape[0]),
        'n_synthetic_samples': int(synth.shape[0]),
        'n_snps': int(real.shape[1]),
        'allele_frequency_pearson_r': _safe_corr(af_real, af_synth),
        'allele_frequency_mae': float(np.mean(np.abs(af_real - af_synth))),
        'allele_frequency_rmse': float(np.sqrt(np.mean((af_real - af_synth) ** 2))),
        'maf_pearson_r': _safe_corr(maf_real, maf_synth),
        'maf_mae': float(np.mean(np.abs(maf_real - maf_synth))),
        'maf_rmse': float(np.sqrt(np.mean((maf_real - maf_synth) ** 2))),
        'mean_heterozygosity_real': float(het_real.mean()),
        'mean_heterozygosity_synthetic': float(het_synth.mean()),
        'heterozygosity_mean_abs_diff': float(abs(het_real.mean() - het_synth.mean())),
        'ld_adjacent_r2_pearson_r': _safe_corr(ld_real[ld_mask], ld_synth[ld_mask]) if ld_mask.any() else float('nan'),
        'ld_adjacent_r2_mae': float(np.mean(np.abs(ld_real[ld_mask] - ld_synth[ld_mask]))) if ld_mask.any() else float('nan'),
        'ld_pairs_evaluated': int(ld_mask.sum()),
        'pc_rbf_mmd2': mmd2,
        'pc_rbf_mmd_bandwidth': mmd_sigma,
        'pc_adversarial_accuracy_truth': aa['aa_truth'],
        'pc_adversarial_accuracy_synthetic': aa['aa_synthetic'],
        'pc_adversarial_accuracy_ts': aa['aa_ts'],
    }
    arrays = {
        'maf_real': maf_real, 'maf_synth': maf_synth,
        'het_real': het_real, 'het_synth': het_synth,
        'ld_real': ld_real[ld_mask], 'ld_synth': ld_synth[ld_mask],
    }
    return metrics, arrays


def plot_quality_metrics(arrays: dict, plot_dir: str):
    # MAF correspondence
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(arrays['maf_real'], arrays['maf_synth'], s=9, alpha=0.35, edgecolors='none')
    ax.plot([0, 0.5], [0, 0.5], linestyle='--', linewidth=1)
    ax.set_xlabel('Real minor allele frequency')
    ax.set_ylabel('Synthetic minor allele frequency')
    ax.set_title('Per-SNP minor allele frequency fidelity')
    plt.tight_layout()
    path = os.path.join(plot_dir, 'maf_real_vs_synthetic.png')
    plt.savefig(path, dpi=160, bbox_inches='tight'); plt.close()
    print(f"  Metric plot saved: {path}", flush=True)

    if len(arrays['ld_real']):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(arrays['ld_real'], arrays['ld_synth'], s=14, alpha=0.4, edgecolors='none')
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        ax.set_xlabel('Real adjacent-pair LD $r^2$')
        ax.set_ylabel('Synthetic adjacent-pair LD $r^2$')
        ax.set_title('Local LD fidelity (sampled adjacent SNP pairs)')
        plt.tight_layout()
        path = os.path.join(plot_dir, 'ld_r2_real_vs_synthetic.png')
        plt.savefig(path, dpi=160, bbox_inches='tight'); plt.close()
        print(f"  Metric plot saved: {path}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arrays['het_real'], bins=30, alpha=0.55, density=True, label='real')
    ax.hist(arrays['het_synth'], bins=30, alpha=0.55, density=True, label='synthetic')
    ax.set_xlabel('Per-sample heterozygosity fraction')
    ax.set_ylabel('Density')
    ax.set_title('Heterozygosity distribution')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plot_dir, 'heterozygosity_distribution.png')
    plt.savefig(path, dpi=160, bbox_inches='tight'); plt.close()
    print(f"  Metric plot saved: {path}", flush=True)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(data_input_paths: list,
                 dataset_members: list,
                 ref_input_paths: list,
                 fam_path: str,
                 model_dir: str,
                 n_synthetic: int = 200,
                 output_name: str = "synthetic_genotypes.csv",
                 plot_name: str = "pca_overlay.png",
                 n_components: int = 50,
                 pca_svd_solver: str = 'randomized',
                 pca_random_state: int = 42,
                 seed: int = 42,
                 target_admixture: np.ndarray = None,
                 alpha_concentration: float = 20.0,
                 variance_scale: float = 0.0,
                 wgan_epochs: int = 3000,
                 wgan_latent_dim: int = 32,
                 wgan_cond_dim: int = 3,
                 wgan_hidden_dims: list = None,
                 batch_size: int = 64,
                 wgan_lr: float = 1e-4,
                 wgan_n_critic: int = 5,
                 wgan_lambda_gp: float = 10.0,
                 wgan_adam_beta1: float = 0.0,
                 wgan_adam_beta2: float = 0.9,
                 wgan_adam_eps: float = 1e-8,
                 wgan_weight_decay: float = 0.0,
                 wgan_amsgrad: bool = False,
                 wgan_leaky_relu_slope: float = 0.2,
                 wgan_layernorm_eps: float = 1e-5,
                 wgan_norm_eps: float = 1e-8,
                 val_fraction: float = 0.2,
                 device: str = 'auto',
                 log_every: int = 500,
                 metric_max_ld_pairs: int = 2000,
                 model_backend: str = 'baseline_pca',
                 condition_source: str = 'pca_barycentric',
                 condition_input_paths: dict = None,
                 external_condition_order: tuple = ('AFR', 'EUR', 'AMR'),
                 balanced_sampling: bool = False,
                 pack_size: int = 1,
                 conv_channels: int = 128,
                 conv_blocks: int = 4,
                 checkpoint_metric: str = 'composite',
                 checkpoint_eval_every: int = 50,
                 checkpoint_metric_max_samples: int = 128,
                 checkpoint_metric_max_ld_pairs: int = 500,
                 save_models: list = None,
                 subset_samples: dict = None,
                 sample_name: str = "BAQ",
                 real_color: str = None):
    """Run one model, with one query dataset or a pre-harmonized combination."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    t_total = time.time()

    model_dir = os.path.abspath(model_dir)
    outputs_dir = os.path.join(model_dir, 'outputs')
    plots_dir = os.path.join(model_dir, 'plots')
    metrics_dir = os.path.join(model_dir, 'metrics')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    for d in (outputs_dir, plots_dir, metrics_dir, checkpoints_dir):
        os.makedirs(d, exist_ok=True)

    output_path = os.path.join(outputs_dir, os.path.basename(output_name))
    plot_path = os.path.join(plots_dir, os.path.basename(plot_name))
    loss_plot_path = os.path.join(plots_dir, 'loss_curves_train_validation.png')
    validation_metric_plot_path = os.path.join(plots_dir, 'validation_selection_metric.png')
    history_path = os.path.join(outputs_dir, 'training_history.csv')
    validation_metric_history_path = os.path.join(outputs_dir, 'validation_selection_history.csv')
    checkpoint_paths = {
        'baseline': os.path.join(checkpoints_dir, 'checkpoint_baseline.pt'),
        'first': os.path.join(checkpoints_dir, 'checkpoint_first.pt'),
        'best': os.path.join(checkpoints_dir, 'checkpoint_best.pt'),
        'last': os.path.join(checkpoints_dir, 'checkpoint_last.pt'),
    }
    checkpoint_path = os.path.join(checkpoints_dir, 'conditional_wgan_checkpoint.pt')
    selected_models_root = os.path.join(model_dir, 'saved_model_outputs')
    os.makedirs(selected_models_root, exist_ok=True)
    save_models = list(dict.fromkeys(save_models or ['best']))

    job_id = os.environ.get('SLURM_JOB_ID', '')
    print("\n" + "=" * 88, flush=True)
    print(f"MODEL: {sample_name} | inputs={dataset_members} | folder={model_dir}", flush=True)
    if job_id:
        print(f"SLURM job: {job_id}", flush=True)
    print("=" * 88, flush=True)

    # Step 1: load query dataset(s).
    t0 = time.time()
    print("\n[1/9] Loading query genotype dataset(s)...", flush=True)
    query_dfs = []
    # This run receives exactly one resolved subset choice per dataset.  ``None``
    # means use all rows.  Sample-size sweeps are expanded into independent runs
    # by main() before run_pipeline() is called.
    resolved_subset_samples = {}
    for k, v in (subset_samples or {}).items():
        name = str(k).upper()
        if v is None or (isinstance(v, str) and v.strip().lower() == 'all'):
            resolved_subset_samples[name] = None
        else:
            resolved_subset_samples[name] = int(v)
    subset_samples = resolved_subset_samples
    query_original_sizes = {}
    query_selected_indices = {}
    subset_record = {}
    for name, input_path in zip(dataset_members, data_input_paths):
        df = pd.read_csv(input_path, header=0)
        original_n = int(df.shape[0])
        query_original_sizes[name] = original_n
        requested_n = subset_samples.get(name)
        # One deterministic permutation per dataset makes the sweep nested across
        # independently trained models: for n1 < n2, the n1 selected rows are a
        # strict subset of the n2 selected rows (before sorting back to file order).
        digest = hashlib.sha256(f"{seed}:{name}".encode('utf-8')).digest()
        dataset_seed = int.from_bytes(digest[:8], byteorder='little', signed=False)
        subset_rng = np.random.default_rng(dataset_seed)
        permutation = subset_rng.permutation(original_n)
        if requested_n is not None:
            if requested_n <= 0:
                raise ValueError(f"subset size for {name} must be positive; got {requested_n}.")
            if requested_n > original_n:
                raise ValueError(
                    f"subset size for {name} is {requested_n}, but the dataset has only {original_n} samples."
                )
            selected = np.sort(permutation[:requested_n])
            df = df.iloc[selected].reset_index(drop=True)
            print(
                f"  {name}: nested random subset {requested_n}/{original_n} samples without replacement "
                f"(dataset_seed={dataset_seed})", flush=True
            )
        else:
            selected = np.arange(original_n, dtype=int)
            if name in subset_samples:
                print(f"  {name}: subset choice='all'; using all {original_n} samples", flush=True)
        query_selected_indices[name] = selected.astype(int)
        subset_record[name] = {
            'original_n_samples': original_n,
            'requested_subset': 'all' if requested_n is None else int(requested_n),
            'used_n_samples': int(len(selected)),
            'dataset_subset_seed': int(dataset_seed),
            'nested_permutation_prefix_length': int(len(selected)) if requested_n is not None else None,
            'selected_row_indices_zero_based': selected.astype(int).tolist(),
        }
        query_dfs.append((name, df))
        print(f"  {name}: using {df.shape[0]} samples x {df.shape[1]} SNPs", flush=True)

    subset_indices_path = os.path.join(outputs_dir, 'dataset_subset_indices.json')
    with open(subset_indices_path, 'w') as fh:
        json.dump(subset_record, fh, indent=2)
    print(f"  Dataset subset record saved: {subset_indices_path}", flush=True)

    # Step 2: member-specific reference tensors + harmonized shared columns.
    t0 = time.time()
    print("\n[2/9] Loading member-specific reference genotypes + labels and aligning SNP columns...", flush=True)
    if len(ref_input_paths) != len(dataset_members):
        raise ValueError(
            f"Expected one reference input per dataset member ({len(dataset_members)}), "
            f"got {len(ref_input_paths)}."
        )
    pop_labels = load_population_labels(fam_path)
    ref_dfs = []
    for name, ref_path in zip(dataset_members, ref_input_paths):
        ref_df_member = pd.read_csv(ref_path, header=0)
        ref_dfs.append((name, ref_path, ref_df_member))
        print(
            f"  {name} reference: {ref_df_member.shape[0]} samples x "
            f"{ref_df_member.shape[1]} SNPs <- {ref_path}", flush=True
        )

    # Intersect SNPs across every query tensor and every member-specific reference tensor.
    shared_snps = list(ref_dfs[0][2].columns)
    for _, df in query_dfs:
        shared_set = set(df.columns)
        shared_snps = [s for s in shared_snps if s in shared_set]
    for _, _, rdf in ref_dfs[1:]:
        shared_set = set(rdf.columns)
        shared_snps = [s for s in shared_snps if s in shared_set]
    if not shared_snps:
        raise ValueError('No shared SNP columns across query dataset(s) and their reference tensor(s).')

    print(f"  Shared SNPs across all query/reference inputs: {len(shared_snps):,}", flush=True)
    for name, df in query_dfs:
        dropped = len(df.columns) - len(shared_snps)
        if dropped:
            print(f"  {name}: dropping {dropped:,} SNP columns not shared by every input", flush=True)
    for name, _, rdf in ref_dfs:
        dropped = len(rdf.columns) - len(shared_snps)
        if dropped:
            print(f"  {name} reference: dropping {dropped:,} non-shared SNP columns", flush=True)

    snp_names = shared_snps
    query_sizes = {name: int(df.shape[0]) for name, df in query_dfs}
    query_mats = [df[shared_snps].values.astype(np.float32) for _, df in query_dfs]
    query_geno = np.vstack(query_mats)

    # For a single-dataset job there is exactly one reference. For a combined job,
    # member-specific reference files may differ in SNP coverage, but they must describe
    # the SAME reference individuals in the SAME row order. Verify equality on the shared
    # SNP set before using the first aligned tensor as the common reference panel.
    aligned_refs = [rdf[shared_snps].values.astype(np.float32) for _, _, rdf in ref_dfs]
    ref_geno = aligned_refs[0]
    if len(aligned_refs) > 1:
        for (name, ref_path, _), arr in zip(ref_dfs[1:], aligned_refs[1:]):
            if arr.shape != ref_geno.shape:
                raise ValueError(
                    f"Combined model reference mismatch: {name} reference has aligned shape {arr.shape}, "
                    f"expected {ref_geno.shape}. Combined members must use reference tensors for the same "
                    "reference individuals in the same row order."
                )
            if not np.array_equal(arr, ref_geno):
                mismatch = int(np.sum(arr != ref_geno))
                raise ValueError(
                    f"Combined model reference mismatch: {name} reference differs from the first member's "
                    f"reference at {mismatch:,} aligned genotype entries. Combined members must use reference "
                    "tensors for the same individuals/genotypes; only SNP coverage may differ."
                )
        print("  Combined-model reference check passed: all member references match on shared SNPs.", flush=True)

    del ref_dfs, aligned_refs, query_dfs, query_mats

    def _impute_missing(mat, label):
        miss = mat < 0
        n_miss = int(miss.sum())
        if n_miss:
            col_sum = np.where(miss, 0.0, mat).sum(axis=0)
            col_cnt = (~miss).sum(axis=0)
            col_mean = np.divide(col_sum, col_cnt, out=np.zeros_like(col_sum), where=col_cnt > 0)
            rows, cols = np.where(miss)
            mat[rows, cols] = col_mean[cols]
            print(f"  {label}: imputed {n_miss:,} missing genotypes "
                  f"({n_miss / mat.size:.2%}) to per-SNP means", flush=True)
        else:
            print(f"  {label}: no missing genotypes", flush=True)
        return mat

    query_geno = _impute_missing(query_geno, sample_name)
    ref_geno = _impute_missing(ref_geno, 'reference')

    if len(pop_labels) != ref_geno.shape[0]:
        print(f"  WARNING: fam has {len(pop_labels)} entries but reference tensor has "
              f"{ref_geno.shape[0]} rows -- truncating labels.", flush=True)
        pop_labels = pop_labels[:ref_geno.shape[0]]
    counts = {sp: pop_labels.count(sp) for sp in _SUPERPOPS}
    print(f"  Combined query: {query_geno.shape[0]} samples x {query_geno.shape[1]} SNPs", flush=True)
    print(f"  Reference: {ref_geno.shape[0]} samples x {ref_geno.shape[1]} SNPs [{_elapsed(t0)}]", flush=True)
    print(f"  Labels: AFR={counts['AFR']} EUR={counts['EUR']} AMR={counts['AMR']}", flush=True)

    # Step 3: PCA fit on reference.
    t0 = time.time()
    print("\n[3/9] PCA compression (fit on reference)...", flush=True)
    compressor = PCACompressor(n_components=n_components)
    # Respect the CLI PCA solver/random state while keeping old defaults.
    n_comp = n_components if n_components is not None else min(ref_geno.shape[0] - 1, ref_geno.shape[1], 50)
    compressor.n_components = n_comp
    compressor.pca = PCA(n_components=n_comp, svd_solver=pca_svd_solver, random_state=pca_random_state)
    print(f"  Input shape : {ref_geno.shape[0]} samples x {ref_geno.shape[1]} SNPs")
    print(f"  Retaining   : {n_comp} PCs")
    ref_pcs = compressor.pca.fit_transform(ref_geno)
    var_exp = np.sum(compressor.pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: {var_exp:.2f}%")
    query_pcs = compressor.transform(query_geno)
    pca_checkpoint_path = os.path.join(checkpoints_dir, 'pca_model.npz')
    np.savez_compressed(
        pca_checkpoint_path,
        components=compressor.pca.components_,
        mean=compressor.pca.mean_,
        explained_variance=compressor.pca.explained_variance_,
        explained_variance_ratio=compressor.pca.explained_variance_ratio_,
        singular_values=compressor.pca.singular_values_,
        snp_names=np.asarray(snp_names, dtype=str),
    )
    print(f"  PCA checkpoint saved: {pca_checkpoint_path}", flush=True)
    print(f"  PCA done [{_elapsed(t0)}]", flush=True)

    # Step 4: conditions.
    t0 = time.time()
    print("\n[4/9] Assigning admixture conditions...", flush=True)
    centroids = _ref_centroids(ref_pcs, pop_labels)
    if centroids[0, 0] < centroids[1, 0]:
        compressor.pca.components_[0] *= -1
        ref_pcs[:, 0] *= -1
        query_pcs[:, 0] *= -1
        centroids[:, 0] *= -1
        print("  [INFO] Flipped PC1 sign for consistent orientation (AFR->positive PC1)", flush=True)

    if condition_source == 'pca_barycentric':
        query_admix = estimate_baq_admixture(query_pcs, centroids)
        print("  Query condition source: PCA barycentric projection", flush=True)
    elif condition_source == 'external':
        condition_input_paths = condition_input_paths or {}
        chunks = []
        for member in dataset_members:
            if member not in condition_input_paths:
                raise ValueError(
                    f"condition_source='external' requires a condition file for {member}. "
                    "Use --condition_input NAME=PATH."
                )
            q_full = load_external_conditions(
                condition_input_paths[member], expected_rows=query_original_sizes[member],
                order=external_condition_order,
            )
            q = q_full[query_selected_indices[member]]
            chunks.append(q)
            print(
                f"  Loaded external ancestry conditions for {member}: full={q_full.shape}, used={q.shape}",
                flush=True,
            )
        query_admix = np.vstack(chunks).astype(np.float32)
        print("  Query condition source: external ancestry coefficients", flush=True)
    else:
        raise ValueError(f"Unknown condition_source={condition_source!r}")

    print(f"  {sample_name} mean admixture -- "
          f"AFR:{query_admix[:,0].mean():.3f} "
          f"EUR:{query_admix[:,1].mean():.3f} "
          f"AMR:{query_admix[:,2].mean():.3f}", flush=True)

    label_arr = np.array(pop_labels)
    ref_admix = np.zeros((ref_geno.shape[0], 3), dtype=np.float32)
    for i, sp in enumerate(_SUPERPOPS):
        ref_admix[label_arr == sp, i] = 1.0
    other_mask = (ref_admix.sum(axis=1) == 0)
    ref_admix[other_mask] = 1.0 / 3.0
    all_pcs = np.vstack([query_pcs, ref_pcs]).astype(np.float32)
    all_geno = np.vstack([query_geno, ref_geno]).astype(np.float32)
    all_admix = np.vstack([query_admix, ref_admix]).astype(np.float32)
    query_groups = []
    for member in dataset_members:
        query_groups.extend([f"QUERY_{member}"] * query_sizes[member])
    ref_groups = [f"REF_{label}" for label in pop_labels]
    training_groups = np.asarray(query_groups + ref_groups, dtype=object)
    print(f"  Combined training set: {all_pcs.shape[0]} samples [{_elapsed(t0)}]", flush=True)
    if balanced_sampling:
        unique_groups, group_counts = np.unique(training_groups, return_counts=True)
        print("  Balanced-sampling groups: " +
              ", ".join(f"{g}={n}" for g, n in zip(unique_groups, group_counts)), flush=True)

    # Step 5: train.
    t0 = time.time()
    print("\n[5/9] Training Conditional WGAN-GP...", flush=True)
    print(f"  Backend         : {model_backend}", flush=True)
    print(f"  Condition source: {condition_source}", flush=True)

    common_kwargs = dict(
        latent_dim=wgan_latent_dim, cond_dim=wgan_cond_dim, n_epochs=wgan_epochs,
        batch_size=batch_size, lr=wgan_lr, n_critic=wgan_n_critic,
        lambda_gp=wgan_lambda_gp, adam_beta1=wgan_adam_beta1,
        adam_beta2=wgan_adam_beta2, adam_eps=wgan_adam_eps,
        weight_decay=wgan_weight_decay, amsgrad=wgan_amsgrad,
        leaky_relu_slope=wgan_leaky_relu_slope, val_fraction=val_fraction,
        device=device, seed=seed, log_every=log_every,
    )

    validation_selection_records = []
    checkpoint_callback = _make_validation_checkpoint_callback(
        model_backend=model_backend, all_geno=all_geno, all_admix=all_admix,
        compressor=compressor, n_query_samples=len(query_geno), n_epochs=wgan_epochs,
        seed=seed, eval_every=checkpoint_eval_every,
        max_samples=checkpoint_metric_max_samples,
        max_ld_pairs=checkpoint_metric_max_ld_pairs,
        selection_metric=checkpoint_metric, records=validation_selection_records,
    )
    if val_fraction == 0:
        print(
            f"  Checkpoint selection: metric={checkpoint_metric}, every={checkpoint_eval_every} epoch(s), "
            f"using IN-SAMPLE query training data because val_fraction=0; "
            f"max_comparison_samples={checkpoint_metric_max_samples}.", flush=True
        )
        print(
            "  [IMPORTANT] This score can select an epoch while training on 100% of the data, "
            "but it is not an out-of-sample validation/generalization estimate.", flush=True
        )
    else:
        print(
            f"  Checkpoint selection: metric={checkpoint_metric}, every={checkpoint_eval_every} epoch(s), "
            f"using held-out validation data; max_comparison_samples={checkpoint_metric_max_samples}", flush=True
        )

    if model_backend == 'baseline_pca':
        if balanced_sampling:
            raise ValueError("--balanced_sampling requires --model_backend pca_film or genotype_conv.")
        if pack_size != 1:
            raise ValueError("--pack_size > 1 requires --model_backend pca_film or genotype_conv.")
        model = ConditionalWGAN(
            hidden_dims=wgan_hidden_dims, layernorm_eps=wgan_layernorm_eps,
            norm_eps=wgan_norm_eps, **common_kwargs,
        )
        history = model.fit(all_pcs, all_admix, epoch_callback=checkpoint_callback)
    elif model_backend == 'pca_film':
        model = FiLMConditionalWGAN(
            hidden_dims=wgan_hidden_dims, layernorm_eps=wgan_layernorm_eps,
            norm_eps=wgan_norm_eps, pack_size=pack_size,
            balanced_sampling=balanced_sampling, **common_kwargs,
        )
        history = model.fit(
            all_pcs, all_admix,
            group_labels=training_groups if balanced_sampling else None,
            epoch_callback=checkpoint_callback,
        )
    elif model_backend == 'genotype_conv':
        print("  NOTE: direct SNP-space convolutional training can be memory intensive; "
              "activation memory scales roughly with batch_size x SNP_count x channel_width.", flush=True)
        model = GenotypeConvWGAN(
            n_snps=all_geno.shape[1], channels=conv_channels, n_blocks=conv_blocks,
            pack_size=pack_size, balanced_sampling=balanced_sampling, **common_kwargs,
        )
        history = model.fit(
            all_geno, all_admix,
            group_labels=training_groups if balanced_sampling else None,
            epoch_callback=checkpoint_callback,
        )
    else:
        raise ValueError(f"Unknown model_backend={model_backend!r}")

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Training history saved: {history_path}", flush=True)

    best_epoch = getattr(model, '_best_epoch', None)
    best_score = getattr(model, '_best_score', None)
    baseline_epoch = 0
    last_epoch = int(wgan_epochs)
    first_epoch = 1

    # Resolve semantic labels to actual stored states/epochs.  With val_fraction=0,
    # "best" is selected by the in-sample training composite rather than held-out validation.
    # Only if no checkpoint score could be computed at all does "best" alias the last epoch.
    resolved_model_states = {}
    requested_checkpoint_epochs = {}
    for label in save_models:
        if label == 'baseline':
            resolved_model_states[label] = 'baseline'
            requested_checkpoint_epochs[label] = baseline_epoch
        elif label == 'first':
            resolved_model_states[label] = 'first'
            requested_checkpoint_epochs[label] = first_epoch
        elif label == 'last':
            resolved_model_states[label] = 'last'
            requested_checkpoint_epochs[label] = last_epoch
        elif label == 'best':
            if getattr(model, '_best_state', None) is not None and best_epoch is not None:
                resolved_model_states[label] = 'best'
                requested_checkpoint_epochs[label] = int(best_epoch)
            else:
                resolved_model_states[label] = 'last'
                requested_checkpoint_epochs[label] = last_epoch
                print(
                    "  [CHECKPOINT NOTICE] 'best' was requested but no checkpoint-selection "
                    "score could be computed. 'best' therefore aliases the final epoch for this run.",
                    flush=True,
                )
        else:
            raise ValueError(f"Unknown save_models label: {label!r}")

    grouped_saved_epochs = _group_checkpoint_labels_by_epoch(requested_checkpoint_epochs)
    for epoch, labels in grouped_saved_epochs.items():
        if len(labels) > 1:
            print(
                f"  [CHECKPOINT OVERLAP] {', '.join(labels)} all resolve to epoch {epoch}; "
                "the model weights are identical and one generated-output set will be shared.",
                flush=True,
            )

    # Shared training diagnostics mark every requested saved state.
    plot_loss_history(history, loss_plot_path, checkpoint_epochs=requested_checkpoint_epochs)

    # Save only the requested semantic checkpoint labels. Multiple labels may intentionally
    # point to identical epoch weights (e.g. best == last).
    saved_checkpoint_paths = {}
    for label in save_models:
        path = checkpoint_paths[label]
        model.save_checkpoint(path, state=resolved_model_states[label])
        saved_checkpoint_paths[label] = path

    # Keep the historical compatibility filename, but never introduce an unrequested state:
    # prefer requested best, then requested last, then requested first, then baseline.
    if 'best' in save_models:
        primary_label = 'best'
    elif 'last' in save_models:
        primary_label = 'last'
    elif 'first' in save_models:
        primary_label = 'first'
    else:
        primary_label = 'baseline'
    model.save_checkpoint(checkpoint_path, state=resolved_model_states[primary_label])
    print(
        f"  Compatibility checkpoint uses requested '{primary_label}' weights "
        f"(epoch {requested_checkpoint_epochs[primary_label]}): {checkpoint_path}",
        flush=True,
    )

    if validation_selection_records:
        pd.DataFrame(validation_selection_records).to_csv(validation_metric_history_path, index=False)
        plot_validation_selection_history(
            validation_selection_records, validation_metric_plot_path,
            checkpoint_epochs=requested_checkpoint_epochs,
        )
        evaluation_bases = sorted({str(r.get('evaluation_basis', 'unknown')) for r in validation_selection_records})
        selection_summary = {
            'selection_metric': checkpoint_metric,
            'lower_is_better': True,
            'evaluation_basis': evaluation_bases[0] if len(evaluation_bases) == 1 else evaluation_bases,
            'in_sample_warning': (
                'val_fraction=0: checkpoint selection used query training samples; this is not an out-of-sample generalization estimate.'
                if 'training_in_sample' in evaluation_bases else None
            ),
            'best_epoch': int(best_epoch) if best_epoch is not None else None,
            'best_score': float(best_score) if best_score is not None else None,
            'requested_saved_models': save_models,
            'resolved_saved_epochs': requested_checkpoint_epochs,
            'evaluated_epochs': [int(r['epoch']) for r in validation_selection_records],
            'score_definition': (
                'composite = mean of finite normalized errors: MAF_MAE/0.5, '
                'heterozygosity_abs_diff, LD_r2_MAE, PC_RBF_MMD2/2, '
                'abs(NNAA-0.5)/0.5; each component clipped to [0,1], with undefined '
                'components assigned worst-case error 1.0' if checkpoint_metric == 'composite'
                else 'PC-space RBF MMD^2 on the checkpoint comparison cohort (held-out validation when available, otherwise in-sample query training data)'
            ),
        }
        with open(os.path.join(metrics_dir, 'validation_selection_summary.json'), 'w') as fh:
            json.dump(selection_summary, fh, indent=2, allow_nan=True)
        print(f"  Checkpoint-selection history saved: {validation_metric_history_path}", flush=True)
    print(f"  Training done [{_elapsed(t0)}]", flush=True)

    # Step 6: construct one deterministic inference-condition matrix shared by every
    # requested checkpoint, so first/best/last comparisons differ only in model weights.
    t0 = time.time()
    print(f"\n[6/9] Preparing conditions for {n_synthetic} synthetic samples...", flush=True)
    rng = np.random.default_rng(seed)
    if variance_scale > 0.0:
        if variance_scale <= 1.0:
            mean_query = query_admix.mean(axis=0)
            var_query = query_admix.var(axis=0)
            alpha0_per = mean_query * (1 - mean_query) / np.maximum(var_query, 1e-10) - 1.0
            alpha0 = max(float(np.nanmedian(alpha0_per)), 0.1)
            alpha_fit = mean_query * alpha0
            query_cond = rng.dirichlet(np.maximum(alpha_fit, 0.01), size=n_synthetic).astype(np.float32)
            uniform = rng.dirichlet([1.0, 1.0, 1.0], size=n_synthetic).astype(np.float32)
            t = float(variance_scale)
            cond_infer = ((1.0 - t) * query_cond + t * uniform)
        else:
            sub = max(1.0 - (variance_scale - 1.0) * 0.5, 0.05)
            t = min(float(variance_scale - 1.0), 1.0)
            uniform = rng.dirichlet([1.0, 1.0, 1.0], size=n_synthetic).astype(np.float32)
            corners = rng.dirichlet([sub, sub, sub], size=n_synthetic).astype(np.float32)
            cond_infer = ((1.0 - t) * uniform + t * corners)
        cond_infer = np.clip(cond_infer, 0.0, None)
        cond_infer /= cond_infer.sum(axis=1, keepdims=True)
        cond_infer = cond_infer.astype(np.float32)
        print(f"  Mode: variance_scale={variance_scale:.2f}", flush=True)
    elif target_admixture is None:
        mean_query = query_admix.mean(axis=0)
        var_query = query_admix.var(axis=0)
        alpha0_per = mean_query * (1 - mean_query) / np.maximum(var_query, 1e-10) - 1.0
        alpha0 = max(float(np.nanmedian(alpha0_per)), 0.1)
        alpha_fit = mean_query * alpha0
        cond_infer = rng.dirichlet(np.maximum(alpha_fit, 0.01), size=n_synthetic).astype(np.float32)
        print(f"  Mode: fitted {sample_name} Dirichlet (alpha={np.round(alpha_fit, 2)}, alpha0={alpha0:.2f})", flush=True)
    else:
        target_admixture = np.array(target_admixture, dtype=np.float32)
        target_admixture /= target_admixture.sum()
        cond_infer = sample_target_admixture(
            n_synthetic, target_admixture,
            alpha_concentration=alpha_concentration, seed=seed)
        print(f"  Mode: target admixture AFR={target_admixture[0]:.3f} "
              f"EUR={target_admixture[1]:.3f} AMR={target_admixture[2]:.3f} "
              f"concentration={alpha_concentration:.1f}", flush=True)
    print(f"  Conditions ready [{_elapsed(t0)}]", flush=True)

    # Generate/evaluate once per UNIQUE epoch/state. 'baseline' is epoch 0 and uses the
    # exact random weights captured before any optimizer update. Files are materialized here
    # after training so every saved state uses identical inference conditions/random seeds.
    # If semantic labels overlap (e.g. best==last),
    # they intentionally share this folder to avoid duplicating very large synthetic CSVs.
    saved_model_results = []
    label_to_output_dir = {}
    for epoch, labels in grouped_saved_epochs.items():
        representative_label = labels[0]
        state_name = resolved_model_states[representative_label]
        joined_labels = '_'.join(labels)
        variant_dir = os.path.join(selected_models_root, f"epoch_{epoch:05d}_{joined_labels}")
        variant_outputs_dir = os.path.join(variant_dir, 'outputs')
        variant_plots_dir = os.path.join(variant_dir, 'plots')
        variant_metrics_dir = os.path.join(variant_dir, 'metrics')
        for d in (variant_outputs_dir, variant_plots_dir, variant_metrics_dir):
            os.makedirs(d, exist_ok=True)
        for label in labels:
            label_to_output_dir[label] = variant_dir

        variant_output_path = os.path.join(variant_outputs_dir, os.path.basename(output_name))
        variant_plot_path = os.path.join(variant_plots_dir, os.path.basename(plot_name))
        variant_cond_path = os.path.join(variant_outputs_dir, 'synthetic_conditioning_admixture.csv')

        print("\n" + "-" * 88, flush=True)
        print(
            f"SAVED MODEL OUTPUT: labels={labels} | epoch={epoch} | state={state_name} | folder={variant_dir}",
            flush=True,
        )
        print("-" * 88, flush=True)
        model.activate_checkpoint_state(state_name)

        # Step 7: sample from this checkpoint. Reset RNGs so checkpoints see identical latent
        # draws and discretization randomness, making their output comparison more controlled.
        t_variant = time.time()
        eval_seed = seed + 900001
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        if model_backend == 'genotype_conv':
            synth_geno = model.sample_genotypes(n_synthetic, cond_infer, seed=eval_seed)
            synth_pcs = compressor.transform(synth_geno.astype(np.float32))
        else:
            synth_pcs = model.sample(n_synthetic, cond_infer)
            synth_geno = discretize_genotypes(compressor.inverse_transform(synth_pcs), seed=eval_seed)
        pd.DataFrame(cond_infer, columns=_SUPERPOPS).to_csv(variant_cond_path, index=False)
        print(f"  [7/9] Sampling complete [{_elapsed(t_variant)}]", flush=True)

        n_fixed = int(np.sum(np.var(synth_geno, axis=0) == 0))
        print(f"  Synthetic shape: {synth_geno.shape}", flush=True)
        print(f"  Fixed SNPs: {n_fixed}/{query_geno.shape[1]} ({100*n_fixed/query_geno.shape[1]:.1f}%)", flush=True)
        pd.DataFrame(synth_geno, columns=snp_names).to_csv(variant_output_path, index=False)
        print(f"  Saved genotypes: {variant_output_path}", flush=True)

        # Step 8: PCA overlay for this checkpoint.
        print("  [8/9] Plotting PCA overlay...", flush=True)
        synth_pcs_plot = (synth_pcs if model_backend == 'genotype_conv'
                          else compressor.transform(synth_geno.astype(np.float32)))
        _, query_vertex_indices = _find_baq_vertices(query_pcs, ref_pcs, pop_labels)
        plot_pca_overlay(
            baq_pcs=query_pcs[:, :2],
            synth_pcs=synth_pcs_plot[:, :2],
            var_explained=compressor.pca.explained_variance_ratio_,
            save_path=variant_plot_path,
            ref_pcs=ref_pcs[:, :2],
            ref_pop_labels=pop_labels,
            triangle_vertices=centroids[:, :2],
            baq_vertex_indices=query_vertex_indices,
            target_admixture=target_admixture,
            job_id=job_id,
            seed=seed,
            sample_name=f"{sample_name} [{'/'.join(labels)} epoch {epoch}]",
            real_color=real_color,
        )

        # Step 9: quantitative quality metrics and all quality plots for this checkpoint.
        print("  [9/9] Computing synthetic genotype quality metrics...", flush=True)
        variant_metrics, metric_arrays = compute_quality_metrics(
            query_geno, synth_geno, query_pcs, synth_pcs_plot,
            max_ld_pairs=metric_max_ld_pairs,
        )
        variant_metrics_path = os.path.join(variant_metrics_dir, 'quality_metrics.json')
        with open(variant_metrics_path, 'w') as fh:
            json.dump(variant_metrics, fh, indent=2, allow_nan=True)
        pd.DataFrame([variant_metrics]).to_csv(
            os.path.join(variant_metrics_dir, 'quality_metrics.csv'), index=False)
        plot_quality_metrics(metric_arrays, variant_plots_dir)

        # Copy shared training/selection plots so each selected-model folder is self-contained.
        import shutil
        if os.path.exists(loss_plot_path):
            shutil.copy2(loss_plot_path, os.path.join(variant_plots_dir, os.path.basename(loss_plot_path)))
        if os.path.exists(validation_metric_plot_path):
            shutil.copy2(
                validation_metric_plot_path,
                os.path.join(variant_plots_dir, os.path.basename(validation_metric_plot_path)),
            )

        variant_summary = {
            'labels': labels,
            'epoch': int(epoch),
            'checkpoint_state_used': state_name,
            'checkpoint_files': {label: saved_checkpoint_paths[label] for label in labels},
            'synthetic_genotypes': variant_output_path,
            'synthetic_conditioning_admixture': variant_cond_path,
            'metrics': variant_metrics,
        }
        with open(os.path.join(variant_dir, 'saved_model_output_summary.json'), 'w') as fh:
            json.dump(variant_summary, fh, indent=2, allow_nan=True)
        saved_model_results.append({**variant_summary, 'output_dir': variant_dir})
        print(f"  Completed saved-model outputs [{_elapsed(t_variant)}]", flush=True)

    # Restore the primary requested state in memory before returning.
    model.activate_checkpoint_state(resolved_model_states[primary_label])

    print(f"\n  Total wall time for {sample_name}: {_elapsed(t_total)}", flush=True)
    return {
        'model': sample_name,
        'model_dir': model_dir,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_validation_score': float(best_score) if best_score is not None else None,
        'requested_saved_models': save_models,
        'resolved_saved_epochs': requested_checkpoint_epochs,
        'saved_checkpoints': saved_checkpoint_paths,
        'checkpoint_default': checkpoint_path,
        'saved_model_outputs': saved_model_results,
        'label_to_output_dir': label_to_output_dir,
        'subset_record': subset_record,
    }


# =============================================================================
# Run management / CLI
# =============================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _parse_hidden_dims(value: str) -> list:
    dims = [int(x.strip()) for x in value.split(',') if x.strip()]
    if not dims or any(x <= 0 for x in dims):
        raise argparse.ArgumentTypeError('hidden dims must be comma-separated positive integers')
    return dims


def _parse_subset_samples(value: str) -> dict:
    """Parse per-dataset sample-size sweeps.

    Accepted examples:
      {"BOG": 20}                         -> one BOG job at n=20
      {"BOG": [10, 20, 30, "all"]}      -> four independent BOG jobs
      {"BOG": [10, 20], "COR": [5, 10]} -> two jobs for each separate spec

    ``all`` is case-insensitive and means use every row.  Duplicate choices are
    removed while preserving order.
    """
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                "--subset_samples must be a dictionary, e.g. "
                "\"{'BOG': [10, 20, 30, 'all'], 'COR': [10, 20]}\""
            ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError('--subset_samples must evaluate to a dictionary.')

    out = {}
    for key, raw in parsed.items():
        name = str(key).strip().upper()
        if not name:
            raise argparse.ArgumentTypeError('subset dataset names cannot be empty.')
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        if not values:
            raise argparse.ArgumentTypeError(f'subset sweep for {name} cannot be empty.')

        normalized = []
        seen = set()
        for val in values:
            if isinstance(val, str) and val.strip().lower() == 'all':
                item = 'all'
            else:
                if isinstance(val, bool):
                    raise argparse.ArgumentTypeError(
                        f'subset size for {name} must be a positive integer or \'all\', not bool.'
                    )
                try:
                    n = int(val)
                except Exception as exc:
                    raise argparse.ArgumentTypeError(
                        f'subset choices for {name} must be positive integers or \'all\'.'
                    ) from exc
                try:
                    exact = float(val) == n
                except Exception:
                    exact = True
                if n <= 0 or not exact:
                    raise argparse.ArgumentTypeError(
                        f'subset size for {name} must be a positive integer; got {val!r}.'
                    )
                item = n
            marker = ('all',) if item == 'all' else ('n', int(item))
            if marker not in seen:
                seen.add(marker)
                normalized.append(item)
        out[name] = normalized
    return out


def _expand_subset_jobs(members: list, subset_sweeps: dict) -> list:
    """Expand one dataset specification into independently trained subset jobs.

    For separate specs this simply yields one job per requested size.  For a
    combined spec, choices across members form a Cartesian product.
    """
    explicit = any(m in subset_sweeps for m in members)
    choice_lists = [subset_sweeps.get(m, ['all']) for m in members]
    jobs = []
    for combo in itertools.product(*choice_lists):
        resolved = {
            m: (None if choice == 'all' else int(choice))
            for m, choice in zip(members, combo)
            if m in subset_sweeps
        }
        base = _folder_name_for_members(members)
        if explicit:
            choice_labels = [
                f"{m}_{'all' if choice == 'all' else 'n' + str(int(choice))}"
                for m, choice in zip(members, combo)
            ]
            # Keep the common single-dataset sweep concise (BOG_n20), while
            # combined models retain all member names and sizes explicitly.
            folder_name = choice_labels[0] if len(members) == 1 else (
                f"{base}__" + '__'.join(choice_labels)
            )
        else:
            folder_name = base
        jobs.append({
            'members': list(members),
            'subset_samples': resolved,
            'subset_choices': {m: choice for m, choice in zip(members, combo)},
            'folder_name': folder_name,
        })
    return jobs


def _parse_dataset_input(items: list, option_name: str = '--dataset_input') -> dict:
    """Parse repeated NAME=PATH mappings and reject duplicate dataset names."""
    mapping = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"{option_name} must be NAME=PATH, got: {item}")
        name, p = item.split('=', 1)
        name = name.strip().upper()
        p = p.strip()
        if not name or not p:
            raise ValueError(f"Invalid {option_name}: {item}")
        if name in mapping:
            raise ValueError(
                f"Duplicate {option_name} mapping for {name}. Specify exactly one path per dataset."
            )
        mapping[name] = p
    return mapping


def _parse_dataset_spec(token: str) -> list:
    """Parse BOG, BAQ, {BOG,BAQ}, or BOG+BAQ into a list of members."""
    token = token.strip()
    if token.startswith('{') and token.endswith('}'):
        token = token[1:-1]
        parts = token.split(',')
    elif '+' in token:
        parts = token.split('+')
    else:
        parts = [token]
    members = [p.strip().strip("'\"").upper() for p in parts if p.strip()]
    if not members:
        raise ValueError(f'Invalid dataset specification: {token}')
    return members


def _folder_name_for_members(members: list) -> str:
    if len(members) == 1:
        return members[0]
    return 'combined_' + '_'.join(members)


def _make_run_dir(run_root: str, run_label: str = None) -> tuple:
    now = datetime.now().astimezone()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    suffix = f"_{run_label}" if run_label else ''
    run_name = timestamp + suffix
    root = Path(run_root).expanduser().resolve()
    run_dir = root / run_name
    counter = 2
    while run_dir.exists():
        run_dir = root / f"{run_name}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir), now.isoformat()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Conditional WGAN-GP synthetic genotype generator with run tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group('Dataset selection')
    data.add_argument('--datasets', nargs='+', default=None,
                      help=("One or more model specifications. Plain names run separately, e.g. "
                            "--datasets BOG BAQ. Quote a combined set, e.g. "
                            "--datasets '{BOG,BAQ}' COR."))
    data.add_argument('--dataset_input', action='append', default=[], metavar='NAME=PATH',
                      help='Dataset path mapping; repeat once per named dataset.')
    data.add_argument('--data_input', default=None,
                      help='Legacy single-dataset input path. Use with --sample_name.')
    data.add_argument('--ref_input', action='append', default=[], metavar='NAME=PATH',
                      help=('Reference genotype CSV mapping; repeat once per named dataset, exactly like '
                            '--dataset_input. Each subset size for a dataset reuses that dataset reference.'))
    data.add_argument('--fam', required=True,
                      help='PLINK .fam for reference samples (ancestry labels).')
    data.add_argument('--subset_samples', type=_parse_subset_samples, default={},
                      help=("Per-dataset nested sample-size sweep. Scalars remain supported; lists create "
                            "independently trained models. Example: --subset_samples "
                            "'{\"BOG\": [10,20,30,\"all\"], \"COR\": [10,20]}'. "
                            "Within each dataset, smaller n values are nested prefixes of one deterministic "
                            "random permutation. 'all' uses the complete query dataset. If --subset_samples is omitted, every requested dataset uses all rows. The reference panel "
                            "is never subset by this option."))
    data.add_argument('--bog_baq_harmonized', action='store_true',
                      help=('Required acknowledgement before training a combined BOG+BAQ model. '
                            'This asserts allele/reference orientation and variant identities were '
                            'harmonized upstream; column-name intersection alone is not sufficient.'))

    data.add_argument('--condition_source', choices=['pca_barycentric', 'external'],
                      default='pca_barycentric',
                      help='How query-sample AFR/EUR/AMR conditions are obtained.')
    data.add_argument('--condition_input', action='append', default=[], metavar='NAME=PATH',
                      help='External ancestry coefficients for one dataset; repeatable. Required with --condition_source external.')
    data.add_argument('--external_condition_order', default='AFR,EUR,AMR',
                      help='Column order for headerless ADMIXTURE-like .Q files.')

    run = parser.add_argument_group('Run/output management')
    run.add_argument('--run_root', default='runs', help='Parent directory for timestamped run folders.')
    run.add_argument('--run_label', default=None, help='Optional suffix appended after the timestamp.')
    run.add_argument('--output', default='synthetic_genotypes.csv',
                     help='Synthetic genotype filename within each model outputs/ folder.')
    run.add_argument('--plot', default='pca_overlay.png',
                     help='PCA overlay filename within each model plots/ folder.')
    run.add_argument('--sample_name', default='BAQ',
                     help='Legacy single-dataset population name / plot label.')
    run.add_argument('--real_color', default=None,
                     help='Optional hex color for real query samples.')

    gen = parser.add_argument_group('Generation / PCA hyperparameters')
    gen.add_argument('--n_synthetic', type=int, default=200,
                     help='Number of synthetic samples generated per trained model; independent of training-set size.')
    gen.add_argument('--n_components', type=int, default=50)
    gen.add_argument('--pca_svd_solver', default='randomized',
                     choices=['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'])
    gen.add_argument('--pca_random_state', type=int, default=42)
    gen.add_argument('--seed', type=int, default=42)
    gen.add_argument('--target_admixture', default=None,
                     help="Comma-separated AFR,EUR,AMR target proportions, e.g. '0.6,0.2,0.2'.")
    gen.add_argument('--alpha_concentration', type=float, default=20.0,
                     help='Dirichlet concentration around target_admixture.')
    gen.add_argument('--variance_scale', type=float, default=0.0,
                     help='Spread inference conditions across the ancestry simplex.')

    model = parser.add_argument_group('Conditional WGAN-GP hyperparameters')
    model.add_argument('--wgan_epochs', type=int, default=3000)
    model.add_argument('--wgan_latent_dim', type=int, default=32)
    model.add_argument('--wgan_cond_dim', type=int, default=3,
                       help='Condition dimension; currently must be 3 for AFR/EUR/AMR.')
    model.add_argument('--wgan_hidden_dims', type=_parse_hidden_dims, default=[128, 256, 128],
                       help='Comma-separated hidden widths shared by generator and critic.')
    model.add_argument('--batch_size', type=int, default=64)
    model.add_argument('--wgan_lr', type=float, default=1e-4)
    model.add_argument('--wgan_n_critic', type=int, default=5)
    model.add_argument('--wgan_lambda_gp', type=float, default=10.0)
    model.add_argument('--wgan_adam_beta1', type=float, default=0.0)
    model.add_argument('--wgan_adam_beta2', type=float, default=0.9)
    model.add_argument('--wgan_adam_eps', type=float, default=1e-8)
    model.add_argument('--wgan_weight_decay', type=float, default=0.0)
    model.add_argument('--wgan_amsgrad', action='store_true',
                       help='Use Adam AMSGrad; disabled in the original script.')
    model.add_argument('--wgan_leaky_relu_slope', type=float, default=0.2)
    model.add_argument('--wgan_layernorm_eps', type=float, default=1e-5)
    model.add_argument('--wgan_norm_eps', type=float, default=1e-8)
    model.add_argument('--val_fraction', type=float, default=0.2,
                       help=('Holdout fraction used for validation losses and checkpoint selection. '
                             'Set to 0 to train on all samples; checkpoint selection then uses the '
                             'query training samples in-sample (not an out-of-sample validation estimate).'))
    model.add_argument('--device', default='auto',
                       help="Torch device such as 'auto', 'cpu', 'cuda', or 'cuda:0'.")
    model.add_argument('--log_every', type=int, default=500)
    model.add_argument('--model_backend', choices=['baseline_pca', 'pca_film', 'genotype_conv'],
                       default='baseline_pca',
                       help='Opt-in architecture: baseline PCA MLP, FiLM-conditioned PCA MLP, or direct SNP-space residual Conv1D.')
    model.add_argument('--balanced_sampling', action='store_true',
                       help='Inverse-frequency training-group sampler; supported by pca_film and genotype_conv.')
    model.add_argument('--pack_size', type=int, default=1,
                       help='PacGAN critic pack size. 1 disables packing; 2 is a useful first experiment.')
    model.add_argument('--conv_channels', type=int, default=128,
                       help='Base channel width for genotype_conv.')
    model.add_argument('--conv_blocks', type=int, default=4,
                       help='Number of residual up/downsample blocks for genotype_conv.')

    checkpoint = parser.add_argument_group('Checkpoint selection')
    checkpoint.add_argument('--checkpoint_metric', choices=['composite', 'pc_mmd2'], default='composite',
                            help=('Metric minimized to select checkpoint_best.pt. composite averages bounded normalized '
                                  'MAF, heterozygosity, LD, PC-MMD and NNAA errors.'))
    checkpoint.add_argument('--checkpoint_eval_every', type=int, default=50,
                            help='Evaluate validation synthetic-data fidelity every N epochs (also epoch 1 and final epoch).')
    checkpoint.add_argument('--checkpoint_metric_max_samples', type=int, default=128,
                            help='Maximum held-out validation samples used per checkpoint evaluation; 0 means no cap.')
    checkpoint.add_argument('--checkpoint_metric_max_ld_pairs', type=int, default=500,
                            help='Maximum adjacent SNP pairs used for validation LD fidelity during checkpoint selection.')
    checkpoint.add_argument('--save_models', nargs='+', choices=['baseline', 'first', 'best', 'last'], default=['best'],
                            help=('Which weight states to save AND generate complete synthetic outputs/plots for. '
                                  "'baseline' is the randomly initialized epoch-0 state before any optimizer update. "
                                  'Accepts one or more values, e.g. --save_models baseline first best last. If requested labels '
                                  'resolve to the same epoch, the overlap is logged and one output set is shared.'))

    metrics = parser.add_argument_group('Quality evaluation')
    metrics.add_argument('--metric_max_ld_pairs', type=int, default=2000,
                         help='Maximum sampled adjacent SNP pairs for LD r^2 fidelity metrics.')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.wgan_cond_dim != 3:
        parser.error('--wgan_cond_dim must be 3 because conditions are AFR/EUR/AMR proportions.')
    if not (0.0 <= args.val_fraction < 1.0):
        parser.error('--val_fraction must be in [0, 1).')
    if args.checkpoint_eval_every < 1:
        parser.error('--checkpoint_eval_every must be >= 1.')
    if args.checkpoint_metric_max_samples < 0:
        parser.error('--checkpoint_metric_max_samples must be >= 0.')
    if args.checkpoint_metric_max_ld_pairs < 0:
        parser.error('--checkpoint_metric_max_ld_pairs must be >= 0.')
    if args.n_synthetic <= 0:
        parser.error('--n_synthetic must be > 0.')

    target = None
    if args.target_admixture is not None:
        parts = [float(x) for x in args.target_admixture.split(',')]
        if len(parts) != 3:
            parser.error('--target_admixture must have exactly 3 values.')
        target = np.array(parts, dtype=np.float32)
        if np.any(target < 0) or target.sum() <= 0:
            parser.error('--target_admixture values must be non-negative and sum to > 0.')

    # Resolve model specifications while preserving the legacy single-file CLI.
    input_map = _parse_dataset_input(args.dataset_input, '--dataset_input')
    ref_input_map = _parse_dataset_input(args.ref_input, '--ref_input')
    condition_input_map = _parse_dataset_input(args.condition_input, '--condition_input')
    condition_order = tuple(x.strip().upper() for x in args.external_condition_order.split(','))
    if set(condition_order) != set(_SUPERPOPS) or len(condition_order) != 3:
        parser.error('--external_condition_order must contain AFR,EUR,AMR exactly once.')
    if args.pack_size < 1:
        parser.error('--pack_size must be >= 1.')
    if args.conv_channels < 16 or args.conv_blocks < 1:
        parser.error('--conv_channels must be >=16 and --conv_blocks must be >=1.')
    if args.datasets:
        specs = [_parse_dataset_spec(x) for x in args.datasets]
        requested_names = {m for spec in specs for m in spec}
        missing = sorted(m for m in requested_names if m not in input_map)
        if missing:
            parser.error('Missing --dataset_input mapping(s) for: ' + ', '.join(missing))
        missing_refs = sorted(m for m in requested_names if m not in ref_input_map)
        if missing_refs:
            parser.error('Missing --ref_input mapping(s) for: ' + ', '.join(missing_refs))
        extra_refs = sorted(set(ref_input_map) - requested_names)
        if extra_refs:
            parser.error('--ref_input contains dataset(s) not requested by --datasets: ' + ', '.join(extra_refs))
    elif args.data_input:
        legacy_name = args.sample_name.strip().upper()
        input_map[legacy_name] = args.data_input
        specs = [[legacy_name]]
        if legacy_name not in ref_input_map:
            parser.error(f'Legacy --data_input mode requires --ref_input {legacy_name}=PATH.')
    else:
        parser.error('Use --datasets with --dataset_input NAME=PATH, or legacy --data_input.')

    requested_dataset_names = {m for spec in specs for m in spec}
    unknown_subset_names = sorted(set(args.subset_samples) - requested_dataset_names)
    if unknown_subset_names:
        parser.error(
            '--subset_samples contains dataset(s) not requested by --datasets: ' +
            ', '.join(unknown_subset_names)
        )

    expanded_jobs = []
    for spec in specs:
        expanded_jobs.extend(_expand_subset_jobs(spec, args.subset_samples))
    # Folder names are intended to be stable identifiers for independent models.
    folder_names = [job['folder_name'] for job in expanded_jobs]
    duplicates = sorted({x for x in folder_names if folder_names.count(x) > 1})
    if duplicates:
        parser.error(
            'Expanded dataset/subset specifications produce duplicate model folders: ' +
            ', '.join(duplicates) + '. Remove duplicate --datasets specifications.'
        )

    if args.condition_source == 'external':
        required_conditions = sorted({m for spec in specs for m in spec})
        missing_conditions = [m for m in required_conditions if m not in condition_input_map]
        if missing_conditions:
            parser.error('Missing --condition_input mapping(s) for: ' + ', '.join(missing_conditions))

    run_dir, started_at = _make_run_dir(args.run_root, args.run_label)
    args_record = vars(args).copy()
    args_record['started_at'] = started_at
    args_record['run_dir'] = run_dir
    args_record['resolved_dataset_specs'] = specs
    args_record['resolved_subset_sweeps'] = args.subset_samples
    args_record['resolved_training_jobs'] = expanded_jobs
    args_record['resolved_dataset_inputs'] = input_map
    args_record['resolved_reference_inputs'] = ref_input_map
    args_record['resolved_condition_inputs'] = condition_input_map
    args_record['resolved_external_condition_order'] = condition_order
    args_json_path = os.path.join(run_dir, 'args.json')
    with open(args_json_path, 'w') as fh:
        json.dump(args_record, fh, indent=2)

    log_path = os.path.join(run_dir, 'run.log')
    log_fh = open(log_path, 'w', buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_fh)
    sys.stderr = _Tee(orig_stderr, log_fh)

    manifest = {'run_dir': run_dir, 'started_at': started_at, 'expanded_training_jobs': expanded_jobs, 'models': []}
    try:
        print(f"Run folder: {run_dir}", flush=True)
        print(f"Started at: {started_at}", flush=True)
        print(f"Arguments saved: {args_json_path}", flush=True)
        print(f"Log file: {log_path}", flush=True)
        print("Resolved model specifications: " + ', '.join(str(s) for s in specs), flush=True)
        print(f"Expanded independent training jobs: {len(expanded_jobs)}", flush=True)
        for i, job in enumerate(expanded_jobs, 1):
            print(
                f"  [{i:>2}/{len(expanded_jobs)}] {job['folder_name']} "
                f"subset={job['subset_choices']}", flush=True
            )

        for job_index, job in enumerate(expanded_jobs, 1):
            members = job['members']
            resolved_job_subset = job['subset_samples']
            if {'BOG', 'BAQ'}.issubset(set(members)):
                print("\n[HARMONIZATION CHECK] Combined BOG+BAQ training requested.", flush=True)
                print("  BOG and BAQ must be harmonized upstream for variant identity, reference/alternate "
                      "alleles, strand/orientation and dosage coding before concatenation.", flush=True)
                if not args.bog_baq_harmonized:
                    raise ValueError(
                        "Combined BOG+BAQ model blocked: rerun only after harmonization and pass "
                        "--bog_baq_harmonized to acknowledge it."
                    )
                print("  --bog_baq_harmonized supplied; proceeding with shared-column alignment.", flush=True)

            folder_name = job['folder_name']
            model_dir = os.path.join(run_dir, folder_name)
            os.makedirs(model_dir, exist_ok=False)
            base_sample_name = '+'.join(members)
            subset_text = ', '.join(
                f"{m}={'all' if job['subset_choices'][m] == 'all' else 'n=' + str(job['subset_choices'][m])}"
                for m in members
            )
            sample_name = f"{base_sample_name} [{subset_text}]" if any(
                m in args.subset_samples for m in members
            ) else base_sample_name
            print(
                f"\n[TRAINING JOB {job_index}/{len(expanded_jobs)}] {folder_name} "
                f"-> {sample_name}", flush=True
            )
            model_args_record = args_record.copy()
            model_args_record['model_name'] = sample_name
            model_args_record['dataset_members'] = members
            model_args_record['dataset_paths'] = {m: input_map[m] for m in members}
            model_args_record['reference_paths'] = {m: ref_input_map[m] for m in members}
            model_args_record['resolved_subset_samples_for_model'] = resolved_job_subset
            model_args_record['subset_choices_for_model'] = job['subset_choices']
            model_args_record['training_job_index'] = job_index
            model_args_record['training_job_count'] = len(expanded_jobs)
            with open(os.path.join(model_dir, 'args.json'), 'w') as fh:
                json.dump(model_args_record, fh, indent=2)
            resolved_color = args.real_color if args.real_color else (
                get_real_color(members[0]) if len(members) == 1 else None
            )
            result = run_pipeline(
                data_input_paths=[input_map[m] for m in members],
                dataset_members=members,
                ref_input_paths=[ref_input_map[m] for m in members],
                fam_path=args.fam,
                model_dir=model_dir,
                n_synthetic=args.n_synthetic,
                output_name=args.output,
                plot_name=args.plot,
                n_components=args.n_components,
                pca_svd_solver=args.pca_svd_solver,
                pca_random_state=args.pca_random_state,
                seed=args.seed,
                target_admixture=target,
                alpha_concentration=args.alpha_concentration,
                variance_scale=args.variance_scale,
                wgan_epochs=args.wgan_epochs,
                wgan_latent_dim=args.wgan_latent_dim,
                wgan_cond_dim=args.wgan_cond_dim,
                wgan_hidden_dims=args.wgan_hidden_dims,
                batch_size=args.batch_size,
                wgan_lr=args.wgan_lr,
                wgan_n_critic=args.wgan_n_critic,
                wgan_lambda_gp=args.wgan_lambda_gp,
                wgan_adam_beta1=args.wgan_adam_beta1,
                wgan_adam_beta2=args.wgan_adam_beta2,
                wgan_adam_eps=args.wgan_adam_eps,
                wgan_weight_decay=args.wgan_weight_decay,
                wgan_amsgrad=args.wgan_amsgrad,
                wgan_leaky_relu_slope=args.wgan_leaky_relu_slope,
                wgan_layernorm_eps=args.wgan_layernorm_eps,
                wgan_norm_eps=args.wgan_norm_eps,
                val_fraction=args.val_fraction,
                device=args.device,
                log_every=args.log_every,
                metric_max_ld_pairs=args.metric_max_ld_pairs,
                model_backend=args.model_backend,
                condition_source=args.condition_source,
                condition_input_paths=condition_input_map,
                external_condition_order=condition_order,
                balanced_sampling=args.balanced_sampling,
                pack_size=args.pack_size,
                conv_channels=args.conv_channels,
                conv_blocks=args.conv_blocks,
                checkpoint_metric=args.checkpoint_metric,
                checkpoint_eval_every=args.checkpoint_eval_every,
                checkpoint_metric_max_samples=args.checkpoint_metric_max_samples,
                checkpoint_metric_max_ld_pairs=args.checkpoint_metric_max_ld_pairs,
                save_models=args.save_models,
                subset_samples=resolved_job_subset,
                sample_name=sample_name,
                real_color=resolved_color,
            )
            manifest['models'].append(result)

        manifest['finished_at'] = datetime.now().astimezone().isoformat()
        with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fh:
            json.dump(manifest, fh, indent=2, allow_nan=True)
        print(f"\nRun complete. Manifest: {os.path.join(run_dir, 'run_manifest.json')}", flush=True)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_fh.close()


if __name__ == '__main__':
    main()
