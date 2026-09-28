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
    # Conservative first run (Dirichlet fitted to observed BAQ admixture):
    python conditional_wgan_pipeline.py \
        --data_input   data/BAQ/BAQ_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \
        --ref_input    data/reference/extractedChrAll_common_snps_forReference_qc_ldpruned_unphased_tensor.csv \
        --fam          data/reference/extractedChrAll_common_snps_forReference.fam

    # Generate for a specific state (e.g. 60% AFR / 20% EUR / 20% AMR):
    python conditional_wgan_pipeline.py ... --target_admixture 0.6,0.2,0.2
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from plot_colors import SYNTHETIC_COLOR, get_real_color


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
    """Conditional generator: concat(z, c) → PC scores.

    Default architecture: (32+3=35) → 128 → 256 → 128 → 50
      z: random latent noise vector (latent_dim=32), sampled from N(0,1)
      c: conditioning admixture vector (cond_dim=3), e.g. [0.6, 0.2, 0.2]
      output: 50-dim PC score vector (same space as real BAQ+ref PC scores)
    LayerNorm is used instead of BatchNorm — stable for small batches.
    """

    def __init__(self, latent_dim: int, cond_dim: int,
                 hidden_dims: list, output_dim: int):
        super().__init__()
        layers, prev = [], latent_dim + cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Concatenate noise and condition along feature axis before passing through MLP
        return self.net(torch.cat([z, c], dim=1))


class _CondNetC(nn.Module):
    """Conditional critic: concat(x, c) → score. No BatchNorm (WGAN-GP).

    Default architecture: (50+3=53) → 128 → 256 → 128 → 1
      x: 50-dim PC score vector (real or generated)
      c: conditioning admixture vector (cond_dim=3)
      output: scalar Wasserstein score (unbounded — higher = more "real")
    BatchNorm omitted because the gradient penalty requires per-sample gradients,
    which BatchNorm disrupts by coupling samples within a batch.
    """

    def __init__(self, input_dim: int, cond_dim: int, hidden_dims: list):
        super().__init__()
        layers, prev = [], input_dim + cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Concatenate PC scores and condition so the critic scores realness
        # relative to the conditioning vector — not ancestry-blind
        return self.net(torch.cat([x, c], dim=1))


class ConditionalWGAN:
    """Wasserstein GAN-GP conditioned on 3-dim admixture proportions.

    Trains on PC scores from both BAQ (admixed) and reference (pure ancestry)
    samples.  Each sample carries a conditioning vector c ∈ Δ² (admixture
    simplex).  At inference, c is set to the target state's ancestry profile.
    """

    def __init__(self,
                 latent_dim: int = 32,
                 cond_dim: int = 3,
                 hidden_dims: list = None,
                 n_epochs: int = 3000,
                 batch_size: int = 64,
                 lr: float = 1e-4,
                 n_critic: int = 5,
                 lambda_gp: float = 10.0,
                 device: str = None,
                 seed: int = 42,
                 log_every: int = 500):
        self.latent_dim  = latent_dim
        self.cond_dim    = cond_dim
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.n_critic    = n_critic
        self.lambda_gp   = lambda_gp
        self.device_str  = device
        self.seed        = seed
        self.log_every   = log_every
        self._G = self._C = self._device = None
        self._mean = self._std = None

    def _resolve_device(self) -> torch.device:
        if self.device_str:
            return torch.device(self.device_str)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                           cond: torch.Tensor) -> torch.Tensor:
        n     = real.size(0)
        # Randomly interpolate between a real and a fake sample for each item in batch
        alpha = torch.rand(n, 1, device=self._device).expand_as(real)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        # Compute critic score at the interpolated point
        grads = torch.autograd.grad(
            outputs=self._C(interp, cond), inputs=interp,
            grad_outputs=torch.ones(n, 1, device=self._device),
            create_graph=True, retain_graph=True,
        )[0]
        # Penalise whenever the gradient norm deviates from 1 (Lipschitz constraint)
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def fit(self, pc_scores: np.ndarray, conditions: np.ndarray):
        """Train on combined BAQ + reference PC scores with admixture conditions.

        Args:
            pc_scores:  (n_total, n_components) — BAQ rows followed by ref rows.
            conditions: (n_total, 3) — admixture proportions per sample.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_samples, n_dim = pc_scores.shape
        self._device = self._resolve_device()

        # Normalise PC scores to zero mean / unit std before training —
        # puts all 50 PC dims on the same scale so the network trains evenly.
        self._mean = pc_scores.mean(axis=0).astype(np.float32)
        self._std  = (pc_scores.std(axis=0) + 1e-8).astype(np.float32)
        normed = ((pc_scores - self._mean) / self._std).astype(np.float32)

        print(f"  cWGAN device    : {self._device}", flush=True)
        print(f"  Training data   : {n_samples} samples × {n_dim} PC dims", flush=True)
        print(f"  Cond dim        : {self.cond_dim}", flush=True)
        print(f"  latent={self.latent_dim}  hidden={self.hidden_dims}  "
              f"epochs={self.n_epochs}  bs={self.batch_size}  "
              f"n_critic={self.n_critic}  λ_gp={self.lambda_gp}", flush=True)

        X = torch.from_numpy(normed)
        C = torch.from_numpy(conditions.astype(np.float32))

        bs     = min(self.batch_size, n_samples)
        loader = DataLoader(TensorDataset(X, C),
                            batch_size=bs, shuffle=True,
                            drop_last=(n_samples > bs))

        G = _CondNetG(self.latent_dim, self.cond_dim,
                      self.hidden_dims, n_dim).to(self._device)
        C_net = _CondNetC(n_dim, self.cond_dim,
                          self.hidden_dims).to(self._device)
        opt_G = torch.optim.Adam(G.parameters(),     lr=self.lr, betas=(0.0, 0.9))
        opt_C = torch.optim.Adam(C_net.parameters(), lr=self.lr, betas=(0.0, 0.9))
        G.train(); C_net.train()
        self._C = C_net
        t_train = time.time()

        for epoch in range(1, self.n_epochs + 1):
            g_sum = c_sum = 0.0
            for real_batch, cond_batch in loader:
                real = real_batch.to(self._device)
                cond = cond_batch.to(self._device)
                b    = real.size(0)

                # --- Critic update (run n_critic=5 times per generator step) ---
                # Training the critic more frequently than the generator is the
                # standard WGAN practice — ensures critic is near-optimal before
                # the generator is updated, giving a meaningful gradient signal.
                for _ in range(self.n_critic):
                    z    = torch.randn(b, self.latent_dim, device=self._device)
                    fake = G(z, cond).detach()   # stop gradients from reaching G
                    gp   = self._gradient_penalty(real, fake, cond)
                    # Wasserstein critic loss: maximise E[C(real)] - E[C(fake)]
                    # → equivalently minimise E[C(fake)] - E[C(real)] + λ·GP
                    lC   = C_net(fake, cond).mean() - C_net(real, cond).mean() \
                           + self.lambda_gp * gp
                    opt_C.zero_grad(); lC.backward(); opt_C.step()

                # --- Generator update (once per batch) ---
                # Generator tries to maximise critic score on its own output,
                # i.e. minimise -E[C(G(z, c), c)]
                z  = torch.randn(b, self.latent_dim, device=self._device)
                lG = -C_net(G(z, cond), cond).mean()
                opt_G.zero_grad(); lG.backward(); opt_G.step()
                g_sum += lG.item(); c_sum += lC.item()

            if self.log_every > 0 and epoch % self.log_every == 0:
                nb = max(1, len(loader))
                print(f"  [cWGAN epoch {epoch:>5}/{self.n_epochs}]  "
                      f"G={g_sum/nb:+.4f}  C={c_sum/nb:+.4f}  "
                      f"[{_elapsed(t_train)}]", flush=True)

        G.eval(); C_net.eval()
        self._G = G
        print(f"  cWGAN training complete  [{_elapsed(t_train)}]", flush=True)

    def sample(self, n_samples: int,
               target_admixture: np.ndarray) -> np.ndarray:
        """Generate synthetic PC scores conditioned on target_admixture.

        Args:
            n_samples:        Number of samples to generate.
            target_admixture: (3,) admixture vector [AFR, EUR, AMR] summing to 1,
                              OR (n_samples, 3) per-sample conditioning.
        Returns:
            (n_samples, n_components) float32 PC scores in original scale.
        """
        if self._G is None:
            raise RuntimeError("Call fit() before sample()")
        self._G.eval()

        cond = np.asarray(target_admixture, dtype=np.float32)
        if cond.ndim == 1:
            cond = np.tile(cond, (n_samples, 1))

        with torch.no_grad():
            z    = torch.randn(n_samples, self.latent_dim, device=self._device)
            c_t  = torch.from_numpy(cond).to(self._device)
            # Generator maps (z, c) → normalised PC scores
            norm = self._G(z, c_t).cpu().numpy()
        # De-normalise back to original PC scale (undo the z-score done in fit())
        return (norm * self._std + self._mean).astype(np.float32)


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
# Main Pipeline
# =============================================================================

def run_pipeline(data_input_path: str,
                 ref_input_path: str,
                 fam_path: str,
                 n_synthetic: int = 200,
                 output_path: str = "synthetic_genotypes.csv",
                 plot_path: str = "pca_overlay.png",
                 n_components: int = 50,
                 seed: int = 42,
                 target_admixture: np.ndarray = None,
                 alpha_concentration: float = 20.0,
                 variance_scale: float = 0.0,
                 wgan_epochs: int = 3000,
                 wgan_latent_dim: int = 32,
                 batch_size: int = 64,
                 sample_name: str = "BAQ",
                 real_color: str = None):
    """Full conditional WGAN pipeline.

    Args:
        data_input_path:     Query population genotype CSV (samples × SNPs).
        ref_input_path:      Reference genotype CSV, same SNP columns as query.
                             Must be QC-filtered and LD-pruned to match BAQ.
        fam_path:            PLINK .fam for reference samples (ancestry labels).
        target_admixture:    (3,) [AFR, EUR, AMR] summing to 1.  If None,
                             uses Dirichlet fitted to observed BAQ admixture
                             (conservative — stays near real distribution).
        alpha_concentration: How tightly inference samples cluster around the
                             target.  High (≥20) = conservative; low (~2) = spread.
        variance_scale:      How much to spread inference conditions across the
                             ancestry simplex.  0.0 = tight BAQ-fitted Dirichlet
                             (conservative default).  1.5 = half uniform + half
                             corner-heavy Dirichlet (recommended for spread).
                             These become the actual conditioning vectors fed to the
                             generator, so the genotypes reflect the spread.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    t_total = time.time()

    job_id = os.environ.get('SLURM_JOB_ID', '')
    if job_id:
        def _tag(path):
            base, ext = path.rsplit('.', 1)
            return f"{base}_{job_id}.{ext}"
        output_path = _tag(output_path)
        plot_path   = _tag(plot_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)),   exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load BAQ genotypes
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[1/7] Loading BAQ genotypes...", flush=True)
    baq_df    = pd.read_csv(data_input_path, header=0)
    print(f"  BAQ: {baq_df.shape[0]} samples × {baq_df.shape[1]} SNPs  "
          f"[{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Load reference genotypes + population labels
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[2/7] Loading reference genotypes + labels...", flush=True)
    ref_df = pd.read_csv(ref_input_path, header=0)

    pop_labels = load_population_labels(fam_path)

    # Align to shared SNP columns — QC may drop a small number of SNPs from
    # either dataset; intersecting preserves exact column correspondence.
    shared_snps = list(baq_df.columns.intersection(ref_df.columns))
    if len(shared_snps) != len(baq_df.columns) or len(shared_snps) != len(ref_df.columns):
        dropped_baq = len(baq_df.columns) - len(shared_snps)
        dropped_ref = len(ref_df.columns) - len(shared_snps)
        print(f"  NOTE: aligning to {len(shared_snps)} shared SNPs  "
              f"(dropped {dropped_baq} from BAQ, {dropped_ref} from ref).",
              flush=True)
        baq_df = baq_df[shared_snps]
        ref_df = ref_df[shared_snps]

    snp_names = list(baq_df.columns)
    baq_geno  = baq_df.values.astype(np.float32)
    ref_geno  = ref_df.values.astype(np.float32)
    del baq_df, ref_df

    # vcf_to_unphased_tensor.py encodes a missing call as -1 (see its
    # np.where on allele == -1). PCA below has no missing-value handling, and
    # -1 sits three units under the valid 0-2 dosage range, so leaving them in
    # lets the missingness pattern dominate the principal axes instead of
    # ancestry. Impute each SNP to its observed mean, per dataset. The printed
    # rate is the headline caveat for any plot produced from sparse input.
    def _impute_missing(mat, label):
        miss = mat < 0
        n_miss = int(miss.sum())
        if n_miss:
            col_sum  = np.where(miss, 0.0, mat).sum(axis=0)
            col_cnt  = (~miss).sum(axis=0)
            col_mean = np.divide(col_sum, col_cnt,
                                 out=np.zeros_like(col_sum),
                                 where=col_cnt > 0)
            rows, cols = np.where(miss)
            mat[rows, cols] = col_mean[cols]
            print(f"  {label}: imputed {n_miss:,} missing genotypes "
                  f"({n_miss / mat.size:.2%}) to per-SNP means", flush=True)
        else:
            print(f"  {label}: no missing genotypes", flush=True)
        return mat

    baq_geno = _impute_missing(baq_geno, "query")
    ref_geno = _impute_missing(ref_geno, "reference")

    # After QC the reference may have fewer rows than the .fam file.
    # Truncate labels to match the tensor row count.
    if len(pop_labels) != ref_geno.shape[0]:
        print(f"  WARNING: fam has {len(pop_labels)} entries but reference "
              f"tensor has {ref_geno.shape[0]} rows — truncating labels.",
              flush=True)
        pop_labels = pop_labels[:ref_geno.shape[0]]

    counts = {sp: pop_labels.count(sp) for sp in _SUPERPOPS}
    print(f"  BAQ: {baq_geno.shape[0]} samples × {baq_geno.shape[1]} SNPs", flush=True)
    print(f"  Reference: {ref_geno.shape[0]} samples × {ref_geno.shape[1]} SNPs  "
          f"[{_elapsed(t0)}]", flush=True)
    print(f"  Labels: AFR={counts['AFR']}  EUR={counts['EUR']}  "
          f"AMR={counts['AMR']}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: PCA — fit on reference, project both datasets
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[3/7] PCA compression (fit on reference)...", flush=True)
    compressor = PCACompressor(n_components=n_components)
    # PCA is fit only on the 644 reference samples (larger, balanced across ancestries)
    # so the principal axes reflect global ancestry structure — not BAQ-specific variation.
    # BAQ samples are then projected into this same reference coordinate system.
    ref_pcs    = compressor.fit_transform(ref_geno)
    baq_pcs    = compressor.transform(baq_geno)
    print(f"  PCA done  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Assign admixture conditions
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[4/7] Assigning admixture conditions...", flush=True)

    # Reference centroids: mean PC position of each pure ancestry group (AFR/EUR/AMR).
    # These form the three corners of the ancestry triangle in PC space.
    print("  Computing reference centroids...", flush=True)
    centroids = _ref_centroids(ref_pcs, pop_labels)

    # Enforce consistent PC1 orientation: AFR centroid PC1 > EUR centroid PC1.
    # PCA sign is arbitrary and can flip between runs/datasets. Negating
    # components_[0] + all PC1 scores is exact — inverse_transform is unaffected
    # because the sign change in the component and scores cancel out.
    if centroids[0, 0] < centroids[1, 0]:
        compressor.pca.components_[0] *= -1
        ref_pcs[:, 0] *= -1
        baq_pcs[:, 0] *= -1
        centroids[:, 0] *= -1
        print("  [INFO] Flipped PC1 sign for consistent orientation (AFR→positive PC1)",
              flush=True)

    # BAQ: project each sample onto the ancestry triangle to get its [AFR, EUR, AMR]
    # proportions.  These become the conditioning vectors used during training.
    baq_admix = estimate_baq_admixture(baq_pcs, centroids)
    print(f"  BAQ mean admixture — "
          f"AFR:{baq_admix[:,0].mean():.3f}  "
          f"EUR:{baq_admix[:,1].mean():.3f}  "
          f"AMR:{baq_admix[:,2].mean():.3f}", flush=True)

    # Reference: known pure ancestry → one-hot conditioning vectors
    # AFR → [1,0,0]  EUR → [0,1,0]  AMR → [0,0,1]
    # These are the "ground truth" conditions during training for reference samples.
    label_arr = np.array(pop_labels)
    ref_admix = np.zeros((ref_geno.shape[0], 3), dtype=np.float32)
    for i, sp in enumerate(_SUPERPOPS):
        ref_admix[label_arr == sp, i] = 1.0
    # Samples labeled OTHER get equal weight — they won't disrupt training much
    other_mask = (ref_admix.sum(axis=1) == 0)
    ref_admix[other_mask] = 1.0 / 3.0
    print(f"  Reference conditions assigned  [{_elapsed(t0)}]", flush=True)

    # Stack BAQ and reference into one training set.
    # The model sees admixed samples (BAQ, ~238) alongside pure-ancestry samples
    # (reference, ~644) — the conditioning vector tells it which is which.
    all_pcs    = np.vstack([baq_pcs,   ref_pcs]).astype(np.float32)
    all_admix  = np.vstack([baq_admix, ref_admix]).astype(np.float32)
    print(f"  Combined training set: {all_pcs.shape[0]} samples", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Train Conditional WGAN-GP
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[5/7] Training Conditional WGAN-GP...", flush=True)
    model = ConditionalWGAN(
        latent_dim  = wgan_latent_dim,
        n_epochs    = wgan_epochs,
        batch_size  = batch_size,
        seed        = seed,
    )
    model.fit(all_pcs, all_admix)
    print(f"  Training done  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Sample — determine target admixture for inference
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n[6/7] Generating {n_synthetic} synthetic samples...", flush=True)

    rng = np.random.default_rng(seed)

    if variance_scale > 0.0:
        # Spread inference conditions across the simplex — the generator is
        # conditioned on these vectors so the genotypes reflect the spread.
        # Mirrors the variance_scale logic from pca_synthetic_pipeline.py but
        # applied to the conditioning vectors rather than post-hoc positions.
        if variance_scale <= 1.0:
            # vs ∈ (0, 1]: linearly blend BAQ-fitted Dirichlet → uniform simplex
            # vs=0 → pure BAQ cluster; vs=1 → fully uniform across the triangle
            mean_baq   = baq_admix.mean(axis=0)
            var_baq    = baq_admix.var(axis=0)
            alpha0_per = mean_baq * (1 - mean_baq) / np.maximum(var_baq, 1e-10) - 1.0
            alpha0     = max(float(np.nanmedian(alpha0_per)), 0.1)
            alpha_fit  = mean_baq * alpha0
            baq_cond   = rng.dirichlet(np.maximum(alpha_fit, 0.01),
                                       size=n_synthetic).astype(np.float32)
            uniform    = rng.dirichlet([1.0, 1.0, 1.0],
                                       size=n_synthetic).astype(np.float32)
            t          = float(variance_scale)
            cond_infer = ((1.0 - t) * baq_cond + t * uniform)
        else:
            # vs > 1: drop BAQ-fitted component — blend uniform with corner-heavy Dirichlet.
            # At vs=1.5: sub=0.75, t=0.5 → cond = 0.5*uniform + 0.5*corners
            # Dirichlet([0.75,0.75,0.75]) is corner-heavy (mass pushed toward vertices),
            # so the blend samples across the full simplex including near-pure corners.
            sub        = max(1.0 - (variance_scale - 1.0) * 0.5, 0.05)
            t          = float(variance_scale - 1.0)
            uniform    = rng.dirichlet([1.0, 1.0, 1.0],
                                       size=n_synthetic).astype(np.float32)
            corners    = rng.dirichlet([sub, sub, sub],
                                       size=n_synthetic).astype(np.float32)
            cond_infer = ((1.0 - t) * uniform + t * corners)
        # Re-normalise (blending can push off-simplex by a hair)
        cond_infer = np.clip(cond_infer, 0.0, None)
        cond_infer /= cond_infer.sum(axis=1, keepdims=True)
        cond_infer = cond_infer.astype(np.float32)
        print(f"  Mode: variance_scale={variance_scale:.2f}  "
              f"(spread conditioning across ancestry simplex)", flush=True)
    elif target_admixture is None:
        # Conservative default: Dirichlet fitted to observed BAQ admixture.
        # Stays near the real distribution — good for validation.
        mean_baq   = baq_admix.mean(axis=0)
        var_baq    = baq_admix.var(axis=0)
        alpha0_per = mean_baq * (1 - mean_baq) / np.maximum(var_baq, 1e-10) - 1.0
        alpha0     = max(float(np.nanmedian(alpha0_per)), 0.1)
        alpha_fit  = mean_baq * alpha0
        cond_infer = rng.dirichlet(np.maximum(alpha_fit, 0.01),
                                   size=n_synthetic).astype(np.float32)
        print(f"  Mode: fitted BAQ Dirichlet  "
              f"(α={np.round(alpha_fit, 2)}, α0={alpha0:.2f})", flush=True)
    else:
        target_admixture = np.array(target_admixture, dtype=np.float32)
        target_admixture /= target_admixture.sum()
        cond_infer = sample_target_admixture(
            n_synthetic, target_admixture,
            alpha_concentration=alpha_concentration, seed=seed)
        print(f"  Mode: target admixture  "
              f"AFR={target_admixture[0]:.3f}  "
              f"EUR={target_admixture[1]:.3f}  "
              f"AMR={target_admixture[2]:.3f}  "
              f"concentration={alpha_concentration:.1f}", flush=True)

    synth_pcs = model.sample(n_synthetic, cond_infer)
    print(f"  Sampling done  [{_elapsed(t0)}]", flush=True)

    # ------------------------------------------------------------------
    # Step 7: Inverse PCA → discretize → save
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[7/7] Reconstructing and saving genotypes...", flush=True)
    # inverse_transform: 50 PC scores → 119,495 continuous SNP values via PCA
    # discretize_genotypes: Binomial(2, v/2) sampling → integer dosage 0/1/2
    synth_geno = discretize_genotypes(compressor.inverse_transform(synth_pcs), seed=seed)
    print(f"  Synthetic shape: {synth_geno.shape}  [{_elapsed(t0)}]", flush=True)

    n_fixed = int(np.sum(np.var(synth_geno, axis=0) == 0))
    print(f"  Fixed SNPs: {n_fixed}/{baq_geno.shape[1]} "
          f"({100*n_fixed/baq_geno.shape[1]:.1f}%)", flush=True)

    pd.DataFrame(synth_geno, columns=snp_names).to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}", flush=True)

    # PCA overlay plot — re-project discretized genotypes to verify consistency
    synth_pcs_plot = compressor.transform(synth_geno.astype(np.float32))
    _, baq_vertex_indices = _find_baq_vertices(baq_pcs, ref_pcs, pop_labels)
    plot_pca_overlay(
        baq_pcs=baq_pcs[:, :2],
        synth_pcs=synth_pcs_plot[:, :2],
        var_explained=compressor.pca.explained_variance_ratio_,
        save_path=plot_path,
        ref_pcs=ref_pcs[:, :2],
        ref_pop_labels=pop_labels,
        triangle_vertices=centroids[:, :2],
        baq_vertex_indices=baq_vertex_indices,
        target_admixture=target_admixture,
        job_id=job_id,
        seed=seed,
        sample_name=sample_name,
        real_color=real_color,
    )

    print(f"\n  Total wall time: {_elapsed(t_total)}", flush=True)
    return synth_geno


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conditional WGAN-GP synthetic genotype generator"
    )
    parser.add_argument("--data_input",   required=True,
                        help="Query population genotype CSV (samples × SNPs)")
    parser.add_argument("--ref_input",    required=True,
                        help="Reference genotype CSV — same SNP columns as the "
                             "query data, "
                             "QC-filtered and LD-pruned to match")
    parser.add_argument("--fam",          required=True,
                        help="PLINK .fam for reference samples (ancestry labels)")
    parser.add_argument("--n_synthetic",  type=int, default=200)
    parser.add_argument("--output",       default="synthetic_genotypes.csv")
    parser.add_argument("--plot",         default="pca_overlay.png")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--target_admixture", default=None,
                        help="Comma-separated AFR,EUR,AMR proportions for the "
                             "target state, e.g. '0.6,0.2,0.2'.  "
                             "If omitted, uses Dirichlet fitted to BAQ admixture "
                             "(conservative validation mode).")
    parser.add_argument("--alpha_concentration", type=float, default=20.0,
                        help="Dirichlet precision around target_admixture. "
                             "High (≥20) = tight cluster near target (default). "
                             "Low (~2) = spread across triangle.")
    parser.add_argument("--variance_scale", type=float, default=0.0,
                        help="Spread inference conditions across the ancestry simplex. "
                             "0.0 = tight BAQ-fitted Dirichlet (default). "
                             "1.5 = half uniform + half corner-heavy (recommended spread). "
                             "Overrides --alpha_concentration when > 0.")
    parser.add_argument("--wgan_epochs",    type=int, default=3000)
    parser.add_argument("--wgan_latent_dim",type=int, default=32)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--sample_name",   default="BAQ",
                        help="Name of the query population shown in the PCA plot legend.")
    parser.add_argument("--real_color",    default=None,
                        help="Hex color for real sample points in the PCA plot. "
                             "Defaults to the registered color for --sample_name "
                             "(see plot_colors.py) so populations stay visually "
                             "consistent, and distinct from the synthetic color.")

    args = parser.parse_args()

    target = None
    if args.target_admixture is not None:
        parts  = [float(x) for x in args.target_admixture.split(',')]
        target = np.array(parts, dtype=np.float32)
        if len(target) != 3:
            raise ValueError("--target_admixture must have exactly 3 values")

    run_pipeline(
        data_input_path     = args.data_input,
        ref_input_path      = args.ref_input,
        fam_path            = args.fam,
        n_synthetic         = args.n_synthetic,
        output_path         = args.output,
        plot_path           = args.plot,
        n_components        = args.n_components,
        seed                = args.seed,
        target_admixture    = target,
        alpha_concentration = args.alpha_concentration,
        variance_scale      = args.variance_scale,
        wgan_epochs         = args.wgan_epochs,
        wgan_latent_dim     = args.wgan_latent_dim,
        batch_size          = args.batch_size,
        sample_name         = args.sample_name,
        real_color          = args.real_color,
    )
