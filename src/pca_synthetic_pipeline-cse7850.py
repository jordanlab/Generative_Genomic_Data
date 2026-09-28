"""
PCA-Space Synthetic Genotype Generator
=======================================
Pipeline: Load CSV tensor → PCA compress → Generate in PC space → Inverse PCA → Discretize

Generator is modular — swap between GaussianCopula (Gaussian/Dirichlet) and
WGANGenerator without changing anything else in the pipeline.

Usage:
    # Gaussian copula:
    python pca_synthetic_pipeline.py --input genotypes.csv --model gaussian_copula ...

    # WGAN-GP (uses GPU automatically):
    python pca_synthetic_pipeline.py --input genotypes.csv --model wgan ...

    # WGAN with triangle-constrained variance expansion:
    python pca_synthetic_pipeline.py --input genotypes.csv --model wgan --variance_scale 1.0 ...
"""

import argparse
import gc
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# =============================================================================
# Population helpers
# =============================================================================

_POP_TO_SUPERPOP = {
    # African
    'ESN': 'AFR', 'GWD': 'AFR', 'LWK': 'AFR', 'MSL': 'AFR',
    'YRI': 'AFR', 'ACB': 'AFR', 'ASW': 'AFR',
    # European
    'CEU': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR',
    # Indigenous American
    'Kar': 'AMR', 'Karitiana': 'AMR',
    'May': 'AMR', 'Maya':      'AMR',
    'CLM': 'AMR', 'MXL': 'AMR', 'PEL': 'AMR', 'PUR': 'AMR',
    'Pim': 'AMR', 'Pima':  'AMR',
    'Sur': 'AMR', 'Surui': 'AMR',
    # East Asian
    'CDX': 'EAS', 'CHB': 'EAS', 'CHS': 'EAS', 'JPT': 'EAS', 'KHV': 'EAS',
    # South Asian
    'BEB': 'SAS', 'GIH': 'SAS', 'ITU': 'SAS', 'PJL': 'SAS', 'STU': 'SAS',
}
_SUPERPOPS = ['AFR', 'EUR', 'AMR', 'EAS', 'SAS']

# Reference population colors
_POP_COLORS = {
    'AFR': '#2166AC',   # blue
    'EUR': '#E87D00',   # orange
    'AMR': '#89CFF0',   # light blue (indigenous American)
    'EAS': '#33A02C',   # green (East Asian)
    'SAS': '#984EA3',   # purple-ish (South Asian)
}

# Human-readable labels for ancestry-pole samples
_VERTEX_LABELS = {
    'AFR': 'Most\nAfrican',
    'EUR': 'Most\nEuropean',
    'AMR': 'Most\nIndigenous Am.',
    'EAS': 'Most\nEast Asian',
    'SAS': 'Most\nSouth Asian',
}


def load_population_labels(fam_path: str) -> list:
    """Parse a PLINK .fam file and return one superpop label per sample."""
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
    A   = np.column_stack([v0 - v2, v1 - v2])
    X   = pc_scores - v2
    ATA = A.T @ A
    XA  = X @ A
    alpha01 = np.linalg.solve(ATA, XA.T).T
    alpha2  = 1.0 - alpha01[:, 0] - alpha01[:, 1]
    return np.column_stack([alpha01, alpha2[:, np.newaxis]])


def _find_baq_vertices(baq_pcs: np.ndarray,
                        ref_pcs: np.ndarray,
                        pop_labels: list) -> tuple:
    """For each superpop, find the BAQ sample closest to that reference centroid.

    Returns:
        vertices:       (3, n_components) PC vectors — AFR, EUR, AMR rows.
        vertex_indices: dict  superpop → BAQ sample index (for plot labeling).
    """
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
        print(f"  {sp} vertex → BAQ sample #{idx}  "
              f"(dist to {sp} centroid: {dists[idx]:.3f})", flush=True)
    return np.array(vertex_pcs), vertex_indices


def _find_ref_centroids(ref_pcs: np.ndarray,
                         pop_labels: list) -> np.ndarray:
    """Compute AFR / EUR / AMR reference population centroids in PC space.

    Using reference centroids (rather than extreme BAQ samples) as the
    generation triangle vertices spans the full ancestry space.  Convex
    combinations of these centroids can produce samples more ancestrally
    extreme than any observed BAQ individual, removing the PC2 floor that
    arises when BAQ-extreme samples are used as vertices.

    Returns:
        (3, n_components) float32 array — rows ordered AFR, EUR, AMR.
    """
    labels   = np.array(pop_labels)
    vertices = []
    for sp in _SUPERPOPS:
        mask     = labels == sp
        centroid = ref_pcs[mask].mean(axis=0)
        vertices.append(centroid)
        print(f"  {sp} centroid ({mask.sum():>3d} samples): "
              f"PC1={centroid[0]:+.2f}  PC2={centroid[1]:+.2f}", flush=True)
    return np.array(vertices, dtype=np.float32)


# =============================================================================
# Generator Interface
# =============================================================================

class BaseGenerator(ABC):
    """Abstract base — implement fit() and sample() to swap in any generator."""

    @abstractmethod
    def fit(self, pc_scores: np.ndarray):
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """Returns (n_samples, n_components) array of synthetic PC scores."""
        pass


# =============================================================================
# Gaussian Copula generators  (Gaussian + Dirichlet admixture)
# =============================================================================

class GaussianGenerator(BaseGenerator):
    """Truncated multivariate Gaussian baseline in PC space."""

    def __init__(self, regularization: float = 1e-6, variance_scale: float = 1.0):
        self.regularization = regularization
        self.variance_scale = variance_scale
        self.mean = self.std = self.bounds_lo = self.bounds_hi = None

    def fit(self, pc_scores: np.ndarray):
        self.mean      = np.mean(pc_scores, axis=0)
        self.std       = np.std(pc_scores, axis=0) + self.regularization
        self.bounds_lo = np.min(pc_scores, axis=0)
        self.bounds_hi = np.max(pc_scores, axis=0)
        print(f"  Fitted Gaussian: {self.mean.shape[0]} components, "
              f"variance_scale={self.variance_scale:.2f}", flush=True)

    def sample(self, n_samples: int) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Call fit() before sample()")
        scaled_std = self.std * self.variance_scale
        a = (self.bounds_lo - self.mean) / scaled_std
        b = (self.bounds_hi - self.mean) / scaled_std
        samples = truncnorm.rvs(
            a[:, np.newaxis], b[:, np.newaxis],
            loc=self.mean[:, np.newaxis], scale=scaled_std[:, np.newaxis],
            size=(len(self.mean), n_samples)
        ).T
        return samples.astype(np.float32)


class DirichletGenerator(BaseGenerator):
    """3-way admixture generator using Dirichlet-distributed mixing proportions."""

    def __init__(self, vertices: np.ndarray, variance_scale: float = 1.0):
        self.vertices       = vertices
        self.variance_scale = variance_scale
        self.alpha_params   = None

    def fit(self, pc_scores: np.ndarray):
        bary = _simplex_project_batch(pc_scores, self.vertices)
        bary = np.clip(bary, 1e-6, None)
        bary /= bary.sum(axis=1, keepdims=True)
        mean_b = bary.mean(axis=0)
        var_b  = bary.var(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha0_per = mean_b * (1.0 - mean_b) / np.maximum(var_b, 1e-10) - 1.0
        alpha0 = max(float(np.nanmedian(alpha0_per)), 0.1)
        self.alpha_params = (mean_b * alpha0).astype(np.float64)
        print(f"  Fitted Dirichlet: α={np.round(self.alpha_params, 2)}, "
              f"α0={alpha0:.2f}, variance_scale={self.variance_scale:.2f}", flush=True)
        print(f"  Mean admixture — AFR:{mean_b[0]:.3f}  EUR:{mean_b[1]:.3f}  "
              f"AMR:{mean_b[2]:.3f}", flush=True)

    def sample(self, n_samples: int) -> np.ndarray:
        if self.alpha_params is None:
            raise RuntimeError("Call fit() before sample()")
        scaled_alpha = np.maximum(self.alpha_params / self.variance_scale, 0.01)
        mix_props = np.random.dirichlet(scaled_alpha, size=n_samples)
        return (mix_props @ self.vertices).astype(np.float32)


# =============================================================================
# Admixture Interpolation generator
# =============================================================================

class AdmixtureInterpolationGenerator(BaseGenerator):
    """Generates PC scores by sampling Dirichlet admixture proportions and
    interpolating between the 3 BAQ ancestry-pole PC vectors.

    Unlike WGAN/Gaussian copula (which imitate the observed BAQ distribution),
    this model explicitly covers the interior of the admixture triangle by
    design — every generated sample is a convex combination of the three
    extreme-BAQ vertex vectors, plus a small residual noise term.

    variance_scale semantics:
        0.0 → Dirichlet fitted to BAQ (reproduces observed admixture
               distribution; generates most samples near the BAQ cloud center)
        1.0 → Uniform Dirichlet [1,1,1] (equal coverage of the full triangle)
        2.0 → Dirichlet [0.5,0.5,0.5] (favours corners / ancestrally-distinct
               samples near the triangle edges and tips)

    noise_scale:
        Scales the residual noise added after interpolation.  The residual is
        computed as the per-component std of (real_PC − interpolated_PC) over
        the BAQ training samples.  1.0 = full empirical residual noise.
        0.0 = pure linear interpolation (no within-cluster variation).
    """

    def __init__(self, vertices: np.ndarray, variance_scale: float = 1.0,
                 noise_scale: float = 0.0, seed: int = 42):
        self.vertices      = vertices          # (3, n_components) — AFR, EUR, AMR
        self.variance_scale = variance_scale
        self.noise_scale   = noise_scale
        self.seed          = seed
        self._alpha_fitted = None
        self._residual_std = None

    def fit(self, pc_scores: np.ndarray):
        np.random.seed(self.seed)

        bary = _simplex_project_batch(pc_scores, self.vertices)
        bary = np.clip(bary, 1e-6, None)
        bary /= bary.sum(axis=1, keepdims=True)

        # Method-of-moments Dirichlet fit to BAQ barycentric coordinates
        mean_b = bary.mean(axis=0)
        var_b  = bary.var(axis=0)
        alpha0_per = mean_b * (1.0 - mean_b) / np.maximum(var_b, 1e-10) - 1.0
        alpha0 = max(float(np.nanmedian(alpha0_per)), 0.1)
        self._alpha_fitted = (mean_b * alpha0).astype(np.float64)

        # Residual std: difference between real PC scores and triangle interpolation
        reconstructed      = (bary @ self.vertices).astype(np.float32)
        self._residual_std = (pc_scores - reconstructed).std(axis=0).astype(np.float32)

        print(f"  AdmixtureInterp: α_fitted={np.round(self._alpha_fitted, 3)}, "
              f"α_sum={self._alpha_fitted.sum():.2f}", flush=True)
        print(f"  variance_scale={self.variance_scale:.2f}, "
              f"noise_scale={self.noise_scale:.2f}", flush=True)
        print(f"  Mean admixture — AFR:{mean_b[0]:.3f}  EUR:{mean_b[1]:.3f}  "
              f"AMR:{mean_b[2]:.3f}", flush=True)

    def _effective_alpha(self) -> np.ndarray:
        """Map variance_scale to Dirichlet concentration parameter."""
        vs = self.variance_scale
        if vs <= 1.0:
            # 0 → fitted (BAQ-concentrated); 1 → uniform [1,1,1]
            t = float(vs)
            return ((1.0 - t) * self._alpha_fitted + t * np.ones(3)).astype(np.float64)
        else:
            # Beyond 1: sub-uniform Dirichlet — lower α favours corners/edges
            sub = max(1.0 - (vs - 1.0) * 0.5, 0.05)
            return np.full(3, sub, dtype=np.float64)

    def sample(self, n_samples: int) -> np.ndarray:
        if self._alpha_fitted is None:
            raise RuntimeError("Call fit() before sample()")
        alpha = np.maximum(self._effective_alpha(), 0.05)
        print(f"  Sampling with effective α={np.round(alpha, 3)}", flush=True)

        # Sample admixture proportions from Dirichlet
        mix_props  = np.random.dirichlet(alpha, size=n_samples)       # (n, 3)

        # Interpolate between vertex PC vectors
        pc_samples = (mix_props @ self.vertices).astype(np.float32)   # (n, n_components)

        # Add scaled residual noise for realistic within-cluster variation
        if self.noise_scale > 0.0:
            noise = np.random.randn(
                n_samples, self.vertices.shape[1]).astype(np.float32)
            pc_samples += noise * self._residual_std * self.noise_scale

        return pc_samples


# =============================================================================
# WGAN-GP generator
# =============================================================================

class _WGANNetG(nn.Module):
    """Generator MLP with LayerNorm."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers, prev = [], latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _WGANNetC(nn.Module):
    """Critic MLP — no BatchNorm (WGAN-GP requirement)."""

    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WGANGenerator(BaseGenerator):
    """Wasserstein GAN with gradient penalty (WGAN-GP) generator in PC score space.

    Trains on PCA-compressed genotype vectors, learns the BAQ distribution, and
    generates new PC score vectors that are inverse-transformed back to genotype
    space by the outer pipeline.

    After generation, an optional triangle-constrained variance expansion can be
    applied via run_pipeline()'s variance_scale parameter — see
    _expand_within_triangle().
    """

    def __init__(self,
                 latent_dim: int = 32,
                 hidden_dims: list = None,
                 n_epochs: int = 3000,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 n_critic: int = 5,
                 lambda_gp: float = 10.0,
                 device: str = None,
                 seed: int = 42,
                 log_every: int = 500):
        self.latent_dim  = latent_dim
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.n_critic    = n_critic
        self.lambda_gp   = lambda_gp
        self.device_str  = device
        self.seed        = seed
        self.log_every   = log_every
        self._G = self._device = self._mean = self._std = None

    def _resolve_device(self) -> torch.device:
        if self.device_str:
            return torch.device(self.device_str)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _gradient_penalty(self, C, real: torch.Tensor,
                           fake: torch.Tensor) -> torch.Tensor:
        n     = real.size(0)
        alpha = torch.rand(n, 1, device=self._device).expand_as(real)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        grads = torch.autograd.grad(
            outputs=C(interp), inputs=interp,
            grad_outputs=torch.ones(n, 1, device=self._device),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def fit(self, pc_scores: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_samples, n_dim = pc_scores.shape
        self._device = self._resolve_device()

        self._mean = pc_scores.mean(axis=0).astype(np.float32)
        self._std  = (pc_scores.std(axis=0) + 1e-8).astype(np.float32)
        normed = ((pc_scores - self._mean) / self._std).astype(np.float32)

        print(f"  WGAN device     : {self._device}", flush=True)
        print(f"  PC score space  : {n_samples} samples × {n_dim} dims", flush=True)
        print(f"  latent={self.latent_dim}  hidden={self.hidden_dims}  "
              f"epochs={self.n_epochs}  bs={min(self.batch_size, n_samples)}  "
              f"n_critic={self.n_critic}  λ_gp={self.lambda_gp}", flush=True)

        bs     = min(self.batch_size, n_samples)
        loader = DataLoader(TensorDataset(torch.from_numpy(normed)),
                            batch_size=bs, shuffle=True,
                            drop_last=(n_samples > bs))

        G = _WGANNetG(self.latent_dim, self.hidden_dims, n_dim).to(self._device)
        C = _WGANNetC(n_dim, self.hidden_dims).to(self._device)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        opt_C = torch.optim.Adam(C.parameters(), lr=self.lr, betas=(0.0, 0.9))
        G.train(); C.train()
        t_train = time.time()

        for epoch in range(1, self.n_epochs + 1):
            g_sum = c_sum = 0.0
            for (real_batch,) in loader:
                real = real_batch.to(self._device)
                b    = real.size(0)
                for _ in range(self.n_critic):
                    z    = torch.randn(b, self.latent_dim, device=self._device)
                    fake = G(z).detach()
                    gp   = self._gradient_penalty(C, real, fake)
                    lC   = C(fake).mean() - C(real).mean() + self.lambda_gp * gp
                    opt_C.zero_grad(); lC.backward(); opt_C.step()
                z  = torch.randn(b, self.latent_dim, device=self._device)
                lG = -C(G(z)).mean()
                opt_G.zero_grad(); lG.backward(); opt_G.step()
                g_sum += lG.item(); c_sum += lC.item()

            if self.log_every > 0 and epoch % self.log_every == 0:
                nb = max(1, len(loader))
                print(f"  [WGAN epoch {epoch:>5}/{self.n_epochs}]  "
                      f"G={g_sum/nb:+.4f}  C={c_sum/nb:+.4f}  "
                      f"[{_elapsed(t_train)}]", flush=True)

        G.eval()
        self._G = G
        print(f"  WGAN training complete  [{_elapsed(t_train)}]", flush=True)

    def sample(self, n_samples: int) -> np.ndarray:
        if self._G is None:
            raise RuntimeError("Call fit() before sample()")
        self._G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self._device)
            normed = self._G(z).cpu().numpy()
        return (normed * self._std + self._mean).astype(np.float32)


# =============================================================================
# Triangle-constrained variance expansion  (WGAN post-processing)
# =============================================================================

def _expand_within_triangle(pc_scores: np.ndarray,
                              vertices: np.ndarray,
                              variance_scale: float,
                              seed: int = 42) -> np.ndarray:
    """Expand WGAN-generated PC scores within the BAQ admixture triangle by
    blending barycentric coordinates with Dirichlet draws.

    The old approach (linear expansion from the batch centroid) amplified the
    AFR-heavy WGAN distribution, pushing samples *up* in PC2 instead of toward
    all three corners.  This approach blends toward progressively more-uniform
    (and eventually corner-heavy) distributions, so every vertex is reachable
    at high variance_scale.

    variance_scale semantics:
        0.0  → pure WGAN output projected inside the triangle (no change)
        0.5  → 50 % WGAN  +  50 % Uniform Dirichlet [1,1,1]
        1.0  → fully Uniform Dirichlet [1,1,1] — fills all three corners equally
        1.5  → 50 % Uniform  +  50 % corner-heavy Dirichlet [0.5,0.5,0.5]
        2.0  → fully corner-heavy Dirichlet [0.5,0.5,0.5] — edges and tips

    Args:
        pc_scores:      (n_samples, n_components) WGAN-generated PC scores.
        vertices:       (3, n_components) BAQ vertex PC vectors (AFR, EUR, AMR).
        variance_scale: 0.0–2.0 expansion strength (see above).
        seed:           RNG seed for Dirichlet draws.

    Returns:
        (n_samples, n_components) PC scores guaranteed inside the BAQ triangle.
    """
    if variance_scale == 0.0:
        return pc_scores.copy()

    rng = np.random.default_rng(seed)
    n   = len(pc_scores)

    # Project WGAN outputs into barycentric coords of the BAQ triangle
    bary = _simplex_project_batch(pc_scores, vertices)   # (n, 3)
    bary = np.clip(bary, 0.0, None)
    bary /= bary.sum(axis=1, keepdims=True)

    if variance_scale <= 1.0:
        # Blend WGAN bary coords → Uniform Dirichlet [1,1,1]
        t = float(variance_scale)
        uniform = rng.dirichlet([1.0, 1.0, 1.0], size=n)
        bary_out = (1.0 - t) * bary + t * uniform
    else:
        # Blend Uniform Dirichlet [1,1,1] → corner-heavy Dirichlet [sub,sub,sub].
        # WGAN bary is intentionally dropped here — it is center-heavy by design
        # (it learned the real BAQ distribution) and would cancel out the spreading.
        # Run-to-run variation comes entirely from the per-run seed passed in.
        sub     = max(1.0 - (variance_scale - 1.0) * 0.5, 0.05)
        t       = float(variance_scale - 1.0)          # 0→1 as vs goes 1→2
        uniform = rng.dirichlet([1.0, 1.0, 1.0], size=n)
        corners = rng.dirichlet([sub, sub, sub], size=n)
        bary_out = (1.0 - t) * uniform + t * corners

    # Reconstruct PC scores from blended barycentric coordinates
    return (bary_out @ vertices).astype(np.float32)


# =============================================================================
# PCA Compression & Reconstruction
# =============================================================================

class PCACompressor:
    """PCA compression of genotype matrix and inverse reconstruction."""

    def __init__(self, n_components: int = None):
        self.n_components  = n_components
        self.pca           = None
        self.genotype_mean = None

    def fit_transform(self, genotypes: np.ndarray) -> np.ndarray:
        n_samples, n_snps = genotypes.shape
        if self.n_components is None:
            self.n_components = min(n_samples - 1, n_snps, 50)
        print(f"  Input shape: {n_samples} samples × {n_snps} SNPs")
        print(f"  Retaining {self.n_components} principal components")
        self.genotype_mean = np.mean(genotypes, axis=0)
        self.pca = PCA(n_components=self.n_components,
                       svd_solver='randomized', random_state=42)
        pc_scores = self.pca.fit_transform(genotypes)
        variance_explained = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"  Variance explained by {self.n_components} PCs: {variance_explained:.2f}%")
        return pc_scores

    def inverse_transform(self, pc_scores: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit_transform() before inverse_transform()")
        return self.pca.inverse_transform(pc_scores)


# =============================================================================
# Discretization
# =============================================================================

def discretize_genotypes(continuous_genotypes: np.ndarray) -> np.ndarray:
    """Round continuous genotype values to nearest valid genotype {0, 1, 2}."""
    return np.rint(np.clip(continuous_genotypes, 0.0, 2.0)).astype(np.int8)


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_quality_metrics(real_genotypes: np.ndarray,
                            synthetic_genotypes: np.ndarray) -> dict:
    metrics = {}
    real_af  = np.mean(real_genotypes,      axis=0) / 2.0
    synth_af = np.mean(synthetic_genotypes, axis=0) / 2.0
    metrics["allele_freq_correlation"] = float(np.corrcoef(real_af, synth_af)[0, 1])

    real_counts  = np.stack([np.mean(real_genotypes == g, axis=0)      for g in range(3)])
    synth_counts = np.stack([np.mean(synthetic_genotypes == g, axis=0) for g in range(3)])
    metrics["mean_genotype_freq_error"] = float(np.mean(np.abs(real_counts - synth_counts)))

    n_snps  = real_genotypes.shape[1]
    snp_idx = (np.random.choice(n_snps, min(10_000, n_snps), replace=False)
               if n_snps > 10_000 else np.arange(n_snps))
    real_sub = real_genotypes[:, snp_idx].astype(np.float32)

    n_check   = min(100, synthetic_genotypes.shape[0])
    idx       = np.random.choice(synthetic_genotypes.shape[0], n_check, replace=False)
    synth_sub = synthetic_genotypes[idx][:, snp_idx].astype(np.float32)

    dist_matrix   = cdist(synth_sub, real_sub, metric='hamming')
    min_distances = dist_matrix.min(axis=1)
    metrics["mean_min_hamming_distance"] = float(np.mean(min_distances))
    metrics["min_hamming_distance"]      = float(np.min(min_distances))

    n_real_check   = min(50, real_genotypes.shape[0])
    real_idx       = np.random.choice(real_genotypes.shape[0], n_real_check, replace=False)
    real_check_sub = real_genotypes[real_idx][:, snp_idx].astype(np.float32)
    rr_dist        = cdist(real_check_sub, real_sub, metric='hamming')
    for ii, orig_i in enumerate(real_idx):
        rr_dist[ii, orig_i] = np.inf
    metrics["aats_proxy"] = float(np.mean(min_distances > rr_dist.min(axis=1).mean()))

    return metrics


# =============================================================================
# PCA Overlay Plot  (PC1 vs PC2)
# =============================================================================

_SUPERPOP_LONG = {
    'AFR': 'African',
    'EUR': 'European',
    'AMR': 'Indigenous American',
    'EAS': 'East Asian',
    'SAS': 'South Asian',
}

# Color for the target (admixed) population being simulated — distinct from all
# reference ancestry colors above.
_TARGET_COLOR = '#E63946'   # red


def plot_pca_overlay(real_pcs: np.ndarray,
                     synth_pcs: np.ndarray,
                     var_explained: np.ndarray,
                     save_path: str = "pca_overlay.png",
                     ref_pcs: np.ndarray = None,
                     ref_pop_labels: list = None,
                     baq_vertex_indices: dict = None,
                     triangle_vertices: np.ndarray = None,
                     job_id: str = "",
                     model_label: str = "",
                     variance_scale: float = 0.0,
                     target_label: str = "Target"):
    """Plot PC1 vs PC2 for real target, synthetic, and reference genotypes."""
    var_pct    = var_explained[:2] * 100
    ref_labels = np.array(ref_pop_labels) if ref_pop_labels is not None else None

    fig, ax = plt.subplots(figsize=(8, 7))

    # ---- Reference pure samples (colored by ancestry) ----
    if ref_pcs is not None:
        if ref_labels is not None:
            for sp in _SUPERPOPS:
                mask = ref_labels == sp
                if mask.any():
                    long_name = _SUPERPOP_LONG.get(sp, sp)
                    ax.scatter(ref_pcs[mask, 0], ref_pcs[mask, 1],
                               c=_POP_COLORS[sp], alpha=0.40, s=22,
                               label=long_name, edgecolors='none', zorder=1)
        else:
            ax.scatter(ref_pcs[:, 0], ref_pcs[:, 1],
                       c='#AAAAAA', alpha=0.40, s=22, label='Reference',
                       edgecolors='none', zorder=1)

    # ---- Admixture polygon (reference centroids — generation boundary) ----
    if triangle_vertices is not None:
        pts = np.vstack([triangle_vertices[:, :2],
                         triangle_vertices[0, :2]])   # close the loop
        ax.plot(pts[:, 0], pts[:, 1], color='#333333',
                linestyle='--', linewidth=1.2, alpha=0.6, zorder=4,
                label='Ancestry boundary')
        for i, sp in enumerate(_SUPERPOPS[:len(triangle_vertices)]):
            ax.scatter(triangle_vertices[i, 0], triangle_vertices[i, 1],
                       marker='^', s=90, color=_POP_COLORS.get(sp, '#333333'),
                       edgecolors='black', linewidths=0.7, zorder=6)

    # ---- Real target population samples ----
    ax.scatter(real_pcs[:, 0], real_pcs[:, 1],
               c=_TARGET_COLOR, alpha=0.9, s=55, label=f'Real ({target_label})',
               edgecolors='white', linewidths=0.4, zorder=2)

    # ---- Synthetic samples ----
    ax.scatter(synth_pcs[:, 0], synth_pcs[:, 1],
               c='#7B2FBE', alpha=0.7, s=55, label='Synthetic',
               edgecolors='white', linewidths=0.4, zorder=3)

    # ---- Star markers on the most-extreme target samples ----
    if baq_vertex_indices is not None:
        for sp, idx in baq_vertex_indices.items():
            x, y = real_pcs[idx, 0], real_pcs[idx, 1]
            ax.plot(x, y, marker='*', markersize=16,
                    color=_TARGET_COLOR, markeredgecolor='white',
                    markeredgewidth=0.5, linestyle='none', zorder=7)

    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)", fontsize=12)
    ax.legend(fontsize=9, markerscale=0.9, loc='best',
              framealpha=0.85, edgecolor='#cccccc')

    title_parts = [f"PC1 vs PC2 — {target_label}"]
    if model_label:
        title_parts.append(f"model={model_label}")
    if job_id:
        title_parts.append(f"job {job_id}")
    ax.set_title("  |  ".join(title_parts), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA overlay plot saved to: {save_path}", flush=True)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(input_path: str,
                 n_synthetic: int = 500,
                 output_path: str = "synthetic_genotypes.csv",
                 plot_path: str = "pca_overlay.png",
                 n_components: int = None,
                 seed: int = 42,
                 reference_path: str = None,
                 fam_path: str = None,
                 vcf_path: str = None,
                 target_pop: str = None,
                 pca_all: bool = False,
                 variance_scale: float = 0.0,
                 model: str = 'gaussian_copula',
                 wgan_epochs: int = 3000,
                 wgan_latent_dim: int = 32,
                 admixture_noise_scale: float = 1.0,
                 triangle_mode: str = 'baq_extremes',
                 max_snps: int = None):
    """Full pipeline: load → compress → generate → (expand) → reconstruct → evaluate.

    variance_scale behaviour:
        gaussian_copula          : controls Gaussian/Dirichlet spread (as before).
        wgan                     : controls barycentric expansion applied AFTER
                                   generation.  0 = pure WGAN output.
        admixture_interpolation  : controls Dirichlet concentration.
                                   0 = fitted to BAQ (near-center clustering);
                                   1 = uniform over triangle;
                                   2 = favour corners/edges.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    t_total = time.time()

    job_id = os.environ.get('SLURM_JOB_ID', '')
    if job_id:
        def _insert_job_id(path: str) -> str:
            base, ext = path.rsplit('.', 1)
            return f"{base}_{job_id}.{ext}"
        output_path = _insert_job_id(output_path)
        plot_path   = _insert_job_id(plot_path)
        print(f"  SLURM job ID : {job_id}", flush=True)
        print(f"  Output CSV   : {output_path}", flush=True)
        print(f"  Output plot  : {plot_path}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)),   exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load genotype matrix
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[1/6] Loading genotype matrix...", flush=True)
    ext = input_path.lower().rsplit(".", 1)[-1]
    snp_names = None
    if ext == "pt":
        genotypes = torch.load(input_path, map_location="cpu",
                               weights_only=True).numpy().astype(np.float32)
    elif ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        # pd.read_csv builds per-column bookkeeping that explodes to 100+ GB for
        # CSVs with >1M columns. Parse row-by-row with numpy instead: peak RAM is
        # just the final (n_samples × n_snps) int8 matrix (~1 GB) plus one row
        # at a time during parsing.
        with open(input_path, 'r') as f:
            snp_names = f.readline().rstrip('\n').split(sep)
            n_snps = len(snp_names)
            n_samples = sum(1 for _ in f)
        print(f"  Allocating ({n_samples} × {n_snps}) int8 buffer ...", flush=True)
        genotypes_i8 = np.empty((n_samples, n_snps), dtype=np.int8)
        with open(input_path, 'r') as f:
            f.readline()  # skip header
            for i, line in enumerate(f):
                genotypes_i8[i] = np.fromstring(line, dtype=np.int8, sep=sep)
                if (i + 1) % 100 == 0:
                    print(f"  Parsed {i+1}/{n_samples} samples", flush=True)
        print(f"  Converting int8 → float32 ...", flush=True)
        genotypes = genotypes_i8.astype(np.float32)
        del genotypes_i8
        gc.collect()
    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    n_samples, n_snps = genotypes.shape
    print(f"  Loaded: {n_samples} samples × {n_snps} SNPs  [{_elapsed(t0)}]", flush=True)
    if genotypes.min() < 0 or genotypes.max() > 2:
        print(f"  WARNING: Values outside {{0,1,2}}: "
              f"range=[{genotypes.min()}, {genotypes.max()}]", flush=True)

    # Remove monomorphic SNPs (zero variance) — no PCA signal, only waste RAM
    col_var = np.var(genotypes, axis=0)
    keep = col_var > 0
    n_mono = int((~keep).sum())
    if n_mono > 0:
        print(f"  Filtering {n_mono} monomorphic SNPs "
              f"({n_mono / n_snps * 100:.1f}%) ...", flush=True)
        if snp_names is not None:
            snp_names = np.array(snp_names)[keep].tolist()
        genotypes = genotypes[:, keep]
        n_samples, n_snps = genotypes.shape
        print(f"  After filter: {n_samples} samples × {n_snps} SNPs", flush=True)
        gc.collect()

    # Optional hard cap on SNP count via random subsampling
    if max_snps is not None and n_snps > max_snps:
        rng_sub = np.random.default_rng(seed)
        idx = np.sort(rng_sub.choice(n_snps, max_snps, replace=False))
        print(f"  Subsampling {n_snps} → {max_snps} SNPs ...", flush=True)
        if snp_names is not None:
            snp_names = [snp_names[i] for i in idx]
        genotypes = genotypes[:, idx]
        n_samples, n_snps = genotypes.shape
        gc.collect()

    # ------------------------------------------------------------------
    # Step 1b: Split target vs. reference using VCF sample labels  OR
    #          load reference from a separate file (--reference / --fam).
    # ------------------------------------------------------------------
    ref_genotypes  = None
    vcf_pop_labels = None
    _pca_all_geno  = None   # holds all genotypes when pca_all=True
    _pca_ref_mask  = None

    if vcf_path is not None and reference_path is None:
        t0 = time.time()
        print(f"\n[1b] Splitting samples by population using VCF: {vcf_path}", flush=True)
        _sample_names = []
        with open(vcf_path) as _fh:
            for _line in _fh:
                if _line.startswith('#CHROM'):
                    _sample_names = _line.strip().split('\t')[9:]
                    break

        if len(_sample_names) != n_samples:
            print(f"  WARNING: VCF has {len(_sample_names)} samples but CSV has "
                  f"{n_samples} rows — skipping split", flush=True)
        else:
            _sample_pops = [s.split('_')[0] for s in _sample_names]
            _tpop = (target_pop or
                     os.path.splitext(os.path.basename(input_path))[0].split('_')[1])
            _target_mask = np.array([p == _tpop for p in _sample_pops])
            _ref_mask    = ~_target_mask
            print(f"  Target ({_tpop}): {_target_mask.sum()} samples", flush=True)
            print(f"  Reference:       {_ref_mask.sum()} samples", flush=True)

            vcf_pop_labels = [_POP_TO_SUPERPOP.get(_sample_pops[i], 'OTHER')
                              for i in range(n_samples) if _ref_mask[i]]

            if pca_all:
                # Keep all samples together for PCA; split only for WGAN training.
                # This ensures the PC axes capture the target population's own
                # variation — critical when reference superpops don't span the
                # target's ancestry space (e.g. GIH with no South Asian reference).
                print(f"  PCA mode: all-samples (target + reference combined)",
                      flush=True)
                _pca_all_geno = genotypes          # all samples — used for PCA
                _pca_ref_mask = _ref_mask
            else:
                ref_genotypes = genotypes[_ref_mask]

            genotypes  = genotypes[_target_mask]   # target only — used for WGAN
            n_samples, n_snps = genotypes.shape
            print(f"  Training matrix: {n_samples} × {n_snps}  [{_elapsed(t0)}]",
                  flush=True)

    elif reference_path is not None:
        t0 = time.time()
        print(f"\n[1b] Loading reference samples from: {reference_path}", flush=True)
        ref_ext = reference_path.lower().rsplit(".", 1)[-1]
        if ref_ext not in ("csv", "tsv"):
            raise ValueError(f"Reference file must be .csv or .tsv, got .{ref_ext}")
        sep    = "\t" if ref_ext == "tsv" else ","
        ref_df = pd.read_csv(reference_path, sep=sep, header=0, index_col=False)
        ref_snp_names = list(ref_df.columns)
        ref_raw = ref_df.values.astype(np.float32)
        del ref_df

        if snp_names is not None and ref_snp_names != snp_names:
            print("  Aligning reference SNP columns...", flush=True)
            ref_col_index = {s: i for i, s in enumerate(ref_snp_names)}
            missing = [s for s in snp_names if s not in ref_col_index]
            if missing:
                raise ValueError(f"Reference missing {len(missing)} SNPs. "
                                 f"First: {missing[:5]}")
            ref_genotypes = ref_raw[:, [ref_col_index[s] for s in snp_names]]
        else:
            ref_genotypes = ref_raw
        print(f"  Reference: {ref_genotypes.shape[0]} samples  [{_elapsed(t0)}]",
              flush=True)

    # ------------------------------------------------------------------
    # Step 2: PCA compression
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[2/6] Compressing via PCA...", flush=True)
    compressor = PCACompressor(n_components=n_components)
    ref_pcs    = None

    if _pca_all_geno is not None:
        # All-samples PCA: fit on the full dataset, then split by mask.
        print(f"  Fitting PCA on all {_pca_all_geno.shape[0]} samples "
              f"(target + reference)...", flush=True)
        all_pc_scores = compressor.fit_transform(_pca_all_geno)
        pc_scores = all_pc_scores[~_pca_ref_mask]   # target
        ref_pcs   = all_pc_scores[_pca_ref_mask]    # reference (for plot)
        del _pca_all_geno
        gc.collect()
        print(f"  PCA done — target: {pc_scores.shape}, ref: {ref_pcs.shape}  "
              f"[{_elapsed(t0)}]", flush=True)
    elif ref_genotypes is not None:
        print(f"  Fitting PCA on {ref_genotypes.shape[0]} reference samples...",
              flush=True)
        compressor.fit_transform(ref_genotypes)
        pc_scores = compressor.pca.transform(genotypes)
        ref_pcs   = compressor.pca.transform(ref_genotypes)
        print(f"  Projected {n_samples} target samples into reference space  "
              f"[{_elapsed(t0)}]", flush=True)
    else:
        pc_scores = compressor.fit_transform(genotypes)
        print(f"  PCA done  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 2b: Population labels + BAQ admixture triangle vertices
    # Computed for BOTH models — needed for:
    #   • colored reference plot and extreme-point labels (always)
    #   • Dirichlet generator selection (gaussian_copula only)
    #   • triangle-constrained variance expansion (wgan + variance_scale > 0)
    # ------------------------------------------------------------------
    pop_labels        = None
    triangle_vertices = None
    baq_vertex_indices = None

    if ref_pcs is not None and (fam_path is not None or vcf_pop_labels is not None):
        if fam_path is not None:
            print(f"\n[2b] Parsing population labels from: {fam_path}", flush=True)
            pop_labels = load_population_labels(fam_path)
        else:
            print(f"\n[2b] Using population labels from VCF sample names.", flush=True)
            pop_labels = vcf_pop_labels

        counts = {sp: pop_labels.count(sp) for sp in _SUPERPOPS}
        print(f"  Labels: AFR={counts['AFR']}  EUR={counts['EUR']}  "
              f"AMR={counts['AMR']}  EAS={counts['EAS']}  "
              f"SAS={counts['SAS']}", flush=True)

        if fam_path is not None:
            # Triangle vertices only computed for the fam-file path (requires all
            # three anchor superpops to be present in the reference set).
            print("  Finding BAQ ancestry-pole vertices (for plot labels)...",
                  flush=True)
            baq_vertices_for_plot, baq_vertex_indices = _find_baq_vertices(
                pc_scores, ref_pcs, pop_labels)

            if triangle_mode == 'ref_centroids':
                print("  Triangle mode: ref_centroids  "
                      "(larger than BAQ-extreme triangle)", flush=True)
                triangle_vertices = _find_ref_centroids(ref_pcs, pop_labels)
            else:
                print("  Triangle mode: baq_extremes", flush=True)
                triangle_vertices = baq_vertices_for_plot

    # ------------------------------------------------------------------
    # Step 3: Fit generator in PC space
    # ------------------------------------------------------------------
    t0 = time.time()

    if model == 'wgan':
        print("\n[3/6] Fitting WGAN-GP generator in PC space...", flush=True)
        generator   = WGANGenerator(latent_dim=wgan_latent_dim,
                                    n_epochs=wgan_epochs, seed=seed)
        model_label = 'WGAN-GP'

    elif model == 'admixture_interpolation':
        if triangle_vertices is None:
            raise ValueError(
                "--model admixture_interpolation requires --reference and --fam "
                "(triangle vertices could not be computed)")
        print("\n[3/6] Fitting AdmixtureInterpolation generator...", flush=True)
        generator   = AdmixtureInterpolationGenerator(
            vertices=triangle_vertices,
            variance_scale=variance_scale,
            noise_scale=admixture_noise_scale,
            seed=seed,
        )
        model_label = 'Admixture Interp'

    elif triangle_vertices is not None:
        print("\n[3/6] Fitting Dirichlet generator (3-way admixture)...", flush=True)
        generator   = DirichletGenerator(vertices=triangle_vertices,
                                         variance_scale=variance_scale)
        model_label = 'Gaussian-Copula (Dirichlet)'

    else:
        print("\n[3/6] Fitting Gaussian generator in PC space...", flush=True)
        generator   = GaussianGenerator(regularization=1e-6,
                                        variance_scale=variance_scale)
        model_label = 'Gaussian-Copula'

    generator.fit(pc_scores)
    print(f"  Generator fitted  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Generate synthetic PC scores
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[4/6] Generating {n_synthetic} synthetic samples...", flush=True)
    synthetic_pc_scores = generator.sample(n_synthetic)

    # WGAN post-processing: triangle-constrained variance expansion
    if model == 'wgan' and variance_scale > 0.0:
        if triangle_vertices is not None:
            print(f"  Applying triangle variance expansion "
                  f"(scale={variance_scale:.2f})...", flush=True)
            synthetic_pc_scores = _expand_within_triangle(
                synthetic_pc_scores, triangle_vertices, variance_scale, seed=seed)
        else:
            print("  WARNING: --variance_scale ignored for WGAN — "
                  "triangle requires --reference and --fam.", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Inverse PCA → discretize
    # ------------------------------------------------------------------
    print(f"\n[5/6] Reconstructing genotypes from PC space...", flush=True)
    synthetic_genotypes = discretize_genotypes(
        compressor.inverse_transform(synthetic_pc_scores))
    print(f"  Synthetic shape: {synthetic_genotypes.shape}  [{_elapsed(t0)}]", flush=True)

    n_fixed = int(np.sum(np.var(synthetic_genotypes, axis=0) == 0))
    print(f"  Fixed SNPs: {n_fixed}/{n_snps} ({100*n_fixed/n_snps:.1f}%)", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Quality metrics
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[6/6] Computing quality metrics...", flush=True)
    metrics = compute_quality_metrics(genotypes.astype(np.int8), synthetic_genotypes)
    print(f"  Allele freq correlation : {metrics['allele_freq_correlation']:.4f}")
    print(f"  Mean genotype freq err  : {metrics['mean_genotype_freq_error']:.4f}")
    print(f"  Mean min Hamming dist   : {metrics['mean_min_hamming_distance']:.4f}")
    print(f"  Min Hamming dist        : {metrics['min_hamming_distance']:.4f}")
    print(f"  AATS proxy              : {metrics['aats_proxy']:.4f}")
    print()
    if metrics["allele_freq_correlation"] > 0.95:
        print("    ✓ Allele frequencies well preserved")
    else:
        print("    ✗ Allele frequencies diverging — check PCA reconstruction")
    if metrics["aats_proxy"] < 0.2:
        print("    ✗ Possible memorization (AATS too low)")
    elif metrics["aats_proxy"] > 0.8:
        print("    ✗ Possible underfitting (AATS too high)")
    else:
        print("    ✓ AATS in healthy range (0.2–0.8)")
    print(f"  Metrics done  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Save synthetic genotypes
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n  Saving synthetic genotypes to: {output_path}", flush=True)
    out_ext = output_path.lower().rsplit(".", 1)[-1]
    if out_ext == "pt":
        torch.save(torch.tensor(synthetic_genotypes, dtype=torch.int8), output_path)
    elif out_ext == "csv":
        pd.DataFrame(synthetic_genotypes, columns=snp_names).to_csv(
            output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: .{out_ext}")
    print(f"  Saved  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # PC1 vs PC2 overlay plot
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n  Generating PC1 vs PC2 overlay plot...", flush=True)
    synth_pcs_plot = compressor.pca.transform(synthetic_genotypes.astype(np.float32))

    # Derive a short population label from the input filename (e.g. "CLM")
    target_label = os.path.basename(input_path).split('_')[1] if '_' in os.path.basename(input_path) else os.path.splitext(os.path.basename(input_path))[0]

    plot_pca_overlay(
        real_pcs=pc_scores[:, :2],
        synth_pcs=synth_pcs_plot[:, :2],
        var_explained=compressor.pca.explained_variance_ratio_,
        save_path=plot_path,
        ref_pcs=ref_pcs[:, :2] if ref_pcs is not None else None,
        ref_pop_labels=pop_labels,
        baq_vertex_indices=baq_vertex_indices,
        triangle_vertices=triangle_vertices[:, :2] if triangle_vertices is not None else None,
        job_id=job_id,
        model_label=model_label,
        variance_scale=variance_scale,
        target_label=target_label,
    )
    print(f"  Plot saved  [{_elapsed(t0)}]", flush=True)

    print(f"\n  Total wall time: {_elapsed(t_total)}", flush=True)
    return synthetic_genotypes, metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic genotype generator: Gaussian copula or WGAN-GP"
    )
    parser.add_argument("--input",       required=True,
                        help="Genotype CSV (samples × SNPs, SNP IDs as header)")
    parser.add_argument("--n_synthetic", type=int, default=500)
    parser.add_argument("--output",      default="synthetic_genotypes.csv")
    parser.add_argument("--plot",        default="pca_overlay.png")
    parser.add_argument("--n_components",type=int, default=None,
                        help="PCA components (default: auto, capped at 50)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--reference",   default=None,
                        help="Reference-population CSV (anchors PC axes)")
    parser.add_argument("--fam",         default=None,
                        help="PLINK .fam for reference samples "
                             "(enables ancestry colors, vertex labels, and "
                             "WGAN triangle variance expansion)")
    parser.add_argument("--vcf",         default=None,
                        help="VCF used to generate the input CSV; sample names "
                             "are read from the #CHROM header line to split "
                             "target vs. reference rows automatically")
    parser.add_argument("--target-pop",  default=None,
                        help="Population code for the target (e.g. CLM). "
                             "Inferred from the input filename if omitted.")
    parser.add_argument("--pca-all",     action="store_true",
                        help="Fit PCA on all samples (target + reference) rather "
                             "than reference only. Recommended when the reference "
                             "superpops do not span the target's ancestry space.")
    parser.add_argument("--model",       default="gaussian_copula",
                        choices=["gaussian_copula", "wgan",
                                 "admixture_interpolation"])
    parser.add_argument("--variance_scale", type=float, default=0.0,
                        help="Spread multiplier. "
                             "gaussian_copula: scales Gaussian/Dirichlet std. "
                             "wgan: expands generated points within the BAQ "
                             "admixture triangle (0=no expansion, 1=2× spread, "
                             "2=3× spread). Requires --reference + --fam.")
    parser.add_argument("--wgan_epochs",    type=int, default=3000)
    parser.add_argument("--wgan_latent_dim",type=int, default=32)
    parser.add_argument("--max_snps",       type=int, default=None,
                        help="Randomly subsample to at most this many SNPs "
                             "after monomorphic filtering (default: no limit)")
    parser.add_argument("--admixture_noise_scale", type=float, default=1.0,
                        help="Residual noise scale for admixture_interpolation. "
                             "1.0 = full empirical residual noise; "
                             "0.0 = pure linear interpolation (no noise).")
    parser.add_argument("--triangle", default="baq_extremes",
                        choices=["baq_extremes", "ref_centroids"],
                        help="Vertices of the admixture generation triangle. "
                             "baq_extremes (default): most-extreme BAQ samples "
                             "(original boundary). "
                             "ref_centroids: reference population centroids — "
                             "a larger triangle that lets WGAN samples cross "
                             "the BAQ extreme boundary.")

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        n_synthetic=args.n_synthetic,
        output_path=args.output,
        plot_path=args.plot,
        n_components=args.n_components,
        seed=args.seed,
        reference_path=args.reference,
        fam_path=args.fam,
        vcf_path=args.vcf,
        target_pop=args.target_pop,
        pca_all=args.pca_all,
        variance_scale=args.variance_scale,
        model=args.model,
        wgan_epochs=args.wgan_epochs,
        wgan_latent_dim=args.wgan_latent_dim,
        admixture_noise_scale=args.admixture_noise_scale,
        triangle_mode=args.triangle,
        max_snps=args.max_snps,
    )
