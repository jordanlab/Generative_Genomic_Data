"""
Fully-connected VAE for genomic (SNP) data generation.

Simple architecture: FC encoder → latent z → FC decoder → per-SNP logits.
No CNN, no per-SNP bias, no extra loss terms. Just reconstruction + KL.

This is the same approach as genomic_vae.py but scaled for large feature
counts (100K+ SNPs) with proper CLI, SLURM integration, and a decoder
that outputs (batch, n_classes, input_dim) for direct CrossEntropyLoss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os

GENOTYPES = 3

class GenomicVAE(nn.Module):
    def __init__(self, input_dim, encoder_fc, latent_dim, decoder_fc,
                 n_classes=GENOTYPES, dropout=0.0, final_rank=0):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Encoder: input_dim → encoder_fc[0] → ... → encoder_fc[-1]
        enc_layers = []
        prev = input_dim
        for h in encoder_fc:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        self.mu = nn.Linear(encoder_fc[-1], latent_dim)
        self.logvar = nn.Linear(encoder_fc[-1], latent_dim)

        # Decoder: latent_dim → decoder_fc[0] → ... → decoder_fc[-1] → output
        dec_layers = []
        prev = latent_dim
        for h in decoder_fc:
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h

        output_dim = input_dim * n_classes
        if final_rank > 0:
            # Low-rank factorization: Linear(prev,r) → ReLU → Linear(r, D*3)
            # Reduces params from prev*D*3 to prev*r + r*D*3
            dec_layers.append(nn.Linear(prev, final_rank))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Linear(final_rank, output_dim))
        else:
            dec_layers.append(nn.Linear(prev, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x.float())
        return self.mu(h), torch.clamp(self.logvar(h), min=-20, max=2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        out = self.decoder(z)  # (batch, input_dim * n_classes)
        return out.view(z.size(0), self.n_classes, self.input_dim)  # (batch, C, L)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def generate(self, n, temperature=1.0):
        """Generate from the standard normal prior z ~ N(0,I)."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.mu.out_features, device=device)
            logits = self.decode(z)  # (n, C, L)
            logits = logits.permute(0, 2, 1) / max(temperature, 1e-6)  # (n, L, C)
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(
                probs.reshape(-1, self.n_classes), 1
            ).view(n, self.input_dim)
        return samples.cpu().numpy()

    def generate_from_data(self, train_data, n, temperature=1.0, noise_std=1.0):
        """Generate by sampling z from the aggregate posterior of the training data.

        1. Encode every training sample → collect all mu vectors.
        2. Compute the mean and std of those mu vectors (aggregate posterior).
        3. Sample new z from N(agg_mean, noise_std * agg_std).
        4. Decode and sample genotypes.

        This ensures z values at generation time match what the decoder saw
        during training, even when KL is near zero.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            X = torch.tensor(train_data, dtype=torch.long).to(device)
            # Encode in chunks to avoid OOM
            all_mu = []
            for i in range(0, X.size(0), 32):
                mu_i, _ = self.encode(X[i:i+32])
                all_mu.append(mu_i)
            all_mu = torch.cat(all_mu, dim=0)  # (N_train, latent_dim)

            agg_mean = all_mu.mean(dim=0)   # (latent_dim,)
            agg_std = all_mu.std(dim=0)      # (latent_dim,)
            agg_std = agg_std.clamp(min=1e-6)  # prevent zero-div

            print(f"[INFO] aggregate posterior: mean norm={agg_mean.norm():.4f}, "
                  f"std range=[{agg_std.min():.4f}, {agg_std.max():.4f}]", flush=True)

            # Sample z from aggregate posterior
            eps = torch.randn(n, self.mu.out_features, device=device)
            z = agg_mean.unsqueeze(0) + noise_std * agg_std.unsqueeze(0) * eps

            logits = self.decode(z)  # (n, C, L)
            logits = logits.permute(0, 2, 1) / max(temperature, 1e-6)  # (n, L, C)
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(
                probs.reshape(-1, self.n_classes), 1
            ).view(n, self.input_dim)
        return samples.cpu().numpy()


def vae_loss(logits, x, mu, logvar, beta=1.0,
             kl_free_bits=0.0, kl_target=0.0, kl_target_weight=0.0,
             std_loss_weight=0.0):
    """VAE loss with optional free-bits and KL target (pull-toward).

    - kl_free_bits: minimum KL per latent dimension (prevents individual
      dims from collapsing to the prior).
    - kl_target: desired total KL.  Uses |KL - target| so that:
        KL < target → gradient is -beta → actively pushes KL UP
        KL > target → gradient is +beta → pushes KL down (normal)
      This pulls KL toward the target from both sides.
    """
    # Data may contain -1 for missing genotypes — ignore those positions.
    recon = F.cross_entropy(logits, x.long(), reduction='mean', ignore_index=-1)

    # Per-dimension KL, averaged over the batch, then summed across dims.
    kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    if kl_free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
    kl = kl_per_dim.sum()

    # Pull KL toward target from both sides:
    #   loss = recon + beta * |KL - target|
    # Below target: loss decreases as KL rises → optimizer pushes KL UP
    # Above target: loss decreases as KL falls → optimizer pushes KL down
    if kl_target > 0:
        kl_for_loss = torch.abs(kl - kl_target)
    else:
        kl_for_loss = kl

    # Optional std-matching loss:
    # Compare global std of decoded expected genotype vs real genotype.
    # This helps prevent over-clumped generations.
    if std_loss_weight > 0:
        probs = torch.softmax(logits, dim=1)  # (B, C, L)
        class_vals = torch.arange(
            logits.size(1), device=logits.device, dtype=probs.dtype
        ).view(1, -1, 1)
        x_hat = torch.sum(probs * class_vals, dim=1)  # (B, L), expected dosage

        valid = (x != -1)
        if valid.any():
            x_real = x.float()[valid]
            x_gen = x_hat[valid]
            std_real = torch.std(x_real, unbiased=False)
            std_gen = torch.std(x_gen, unbiased=False)
            std_loss = torch.abs(std_gen - std_real)
        else:
            std_loss = torch.zeros((), device=logits.device)
    else:
        std_loss = torch.zeros((), device=logits.device)

    total_loss = recon + beta * kl_for_loss + std_loss_weight * std_loss
    return total_loss, recon.item(), kl.item(), std_loss.item()


def train(model, data, epochs, batch_size, lr, beta, kl_anneal_epochs, device,
          weight_decay=0.0, kl_free_bits=0.0, kl_target=0.0, kl_target_weight=0.0,
          std_loss_weight=0.0):
    model.to(device)
    X = torch.tensor(data, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X, X)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    print(f"[INFO] train(): {len(ds)} samples, batch_size={batch_size}, "
          f"lr={lr}, beta={beta}, kl_anneal={kl_anneal_epochs}, "
          f"weight_decay={weight_decay}, kl_free_bits={kl_free_bits}, "
            f"kl_target={kl_target}, kl_target_weight={kl_target_weight}, "
            f"std_loss_weight={std_loss_weight}", flush=True)

    history = []
    nan_epochs = 0
    for epoch in range(epochs):
        model.train()
        tot_loss = tot_recon = tot_kl = tot_std = 0.0
        cur_beta = beta * min(1.0, (epoch + 1) / max(kl_anneal_epochs, 1))
        for bx, by in dl:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, mu, logvar = model(bx)
            loss, r, k, s = vae_loss(
                logits, by, mu, logvar, beta=cur_beta,
                kl_free_bits=kl_free_bits,
                kl_target=kl_target,
                kl_target_weight=kl_target_weight,
                std_loss_weight=std_loss_weight,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN/Inf loss at epoch {epoch+1}, skipping batch", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()
            tot_recon += r
            tot_kl += k
            tot_std += s
        n = len(dl)
        avg_loss = tot_loss / n if n > 0 else float('nan')
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Recon: {tot_recon/n:.4f}, KL: {tot_kl/n:.4f}, "
              f"STD: {tot_std/n:.4f}, beta: {cur_beta:.4f}",
              flush=True)
        if np.isnan(avg_loss):
            nan_epochs += 1
            if nan_epochs >= 3:
                print("[ERROR] 3 consecutive NaN epochs, stopping early", flush=True)
                break
        else:
            nan_epochs = 0

    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=100)
    print("[INFO] train(): complete", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description="FC-VAE for genomic SNP data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset', required=True, help='CSV file (rows=samples, cols=SNPs)')
    p.add_argument('--output', default=None, help='Output CSV path')
    p.add_argument('--snp-subset', default=None, help='Text file with SNP names to subset')
    p.add_argument('--encoder-fc', type=lambda s: [int(x) for x in s.split(',')],
                   default=[512, 256, 128],
                   help='Encoder FC hidden dims (comma-separated)')
    p.add_argument('--decoder-fc', type=lambda s: [int(x) for x in s.split(',')],
                   default=[128, 256, 512],
                   help='Decoder FC hidden dims (comma-separated)')
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--final-rank', type=int, default=0,
                   help='If >0, factorize the final decoder layer into two '
                        'smaller layers with this inner dimension. '
                        'Drastically reduces params to prevent memorization.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0, help='KL weight')
    p.add_argument('--kl-anneal-epochs', type=int, default=0,
                   help='Epochs to linearly ramp beta from 0 to --beta')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='L2 regularization on all weights')
    p.add_argument('--n-generate', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--generation-mode', choices=['prior', 'aggregate'],
                   default='aggregate',
                   help="'prior': sample z~N(0,I); "
                        "'aggregate': sample z from encoded training data distribution")
    p.add_argument('--noise-std', type=float, default=1.0,
                   help='Scale factor for spread of z in aggregate mode '
                        '(1.0=match training spread, <1=tighter, >1=wider)')
    p.add_argument('--kl-free-bits', type=float, default=0.0,
                   help='Minimum KL per latent dimension (free-bits). '
                        'E.g. 0.05 ensures every dim contributes >= 0.05 nats.')
    p.add_argument('--kl-target', type=float, default=0.0,
                   help='Desired total KL value. A quadratic penalty pulls '
                        'KL back up when it drops below this. E.g. 0.15')
    p.add_argument('--kl-target-weight', type=float, default=0.0,
                   help='Strength of the soft KL target penalty. '
                        'E.g. 100-500. Only active when --kl-target > 0.')
    p.add_argument('--std-loss-weight', type=float, default=0.0,
                   help='Weight for std-matching loss between real and decoded '
                        'expected genotypes. Helps avoid over-clumped outputs.')
    p.add_argument('--slurm-id', default=None)
    return p


def main():
    args = build_parser().parse_args()

    print(f"[INFO] loading {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset)
    print(f"[INFO] shape: {df.shape}", flush=True)

    if args.snp_subset:
        cols = np.loadtxt(args.snp_subset, dtype=str)
        df = df[cols]
        print(f"[INFO] subset to {df.shape[1]} SNPs", flush=True)

    values = df.values
    columns = df.columns
    input_dim = df.shape[1]

    print(f"[INFO] encoder_fc={args.encoder_fc}, latent_dim={args.latent_dim}, "
          f"decoder_fc={args.decoder_fc}", flush=True)
    model = GenomicVAE(
        input_dim=input_dim,
        encoder_fc=args.encoder_fc,
        latent_dim=args.latent_dim,
        decoder_fc=args.decoder_fc,
        dropout=args.dropout,
        final_rank=args.final_rank,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params: {total_params:,}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}", flush=True)

    train(model, values, args.epochs, args.batch_size, args.lr,
          args.beta, args.kl_anneal_epochs, device,
          weight_decay=args.weight_decay,
          kl_free_bits=args.kl_free_bits,
          kl_target=args.kl_target,
            kl_target_weight=args.kl_target_weight,
            std_loss_weight=args.std_loss_weight)

    print(f"[INFO] generating {args.n_generate} samples (mode={args.generation_mode})", flush=True)
    if args.generation_mode == 'aggregate':
        data = model.generate_from_data(
            values, args.n_generate,
            temperature=args.temperature,
            noise_std=args.noise_std,
        )
    else:
        data = model.generate(args.n_generate, temperature=args.temperature)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.dataset))[0]
        tag = f"_slurm_{args.slurm_id}" if args.slurm_id else ""
        out = os.path.join(os.getcwd(), "results", "gen_data",
                           f"{base}_generated{tag}.csv")
    else:
        out = args.output
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(out, index=False)
    print(f"[INFO] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
