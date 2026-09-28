import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale
import argparse
import os

GENOTYPES = 3

def build_norm_1d(num_channels, norm_type="group", groupnorm_groups=8):
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "group":
        groups = min(groupnorm_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def pick_override(cli_value, preset_value):
    return preset_value if cli_value is None else cli_value


def get_architecture_preset(input_dim, preset_name="auto"):
    presets = {
        "small": {
            "enc_channels": [32, 64, 128],
            "enc_kernel_sizes": [7, 5, 5],
            "enc_strides": [2, 2, 2],
            "enc_paddings": [3, 2, 2],
            "pool_size": 2,
            "adaptive_pool_out": 32,
            "dec_kernel_sizes": [5, 5, 7],
            "dec_strides": [2, 2, 2],
            "dec_paddings": [2, 2, 3],
            "encoder_fc": [512, 128, 32],
            "decoder_fc": [32, 128, 512],
            "latent_dim": 8,
            "norm_type": "group",
            "groupnorm_groups": 8,
            "batch_size": 16,
            "lr": 1e-3,
            "beta": 0.02,
            "kl_anneal_epochs": 20,
        },
        "medium": {
            "enc_channels": [32, 64, 128],
            "enc_kernel_sizes": [9, 7, 5],
            "enc_strides": [4, 2, 2],
            "enc_paddings": [4, 3, 2],
            "pool_size": 2,
            "adaptive_pool_out": 48,
            "dec_kernel_sizes": [5, 7, 9],
            "dec_strides": [2, 2, 4],
            "dec_paddings": [2, 3, 4],
            "encoder_fc": [1024, 256, 64],
            "decoder_fc": [64, 256, 1024],
            "latent_dim": 12,
            "norm_type": "group",
            "groupnorm_groups": 8,
            "batch_size": 8,
            "lr": 5e-4,
            "beta": 0.05,
            "kl_anneal_epochs": 30,
        },
        "large": {
            "enc_channels": [16, 32, 64, 128],
            "enc_kernel_sizes": [9, 7, 5, 5],
            "enc_strides": [4, 4, 2, 2],
            "enc_paddings": [4, 3, 2, 2],
            "pool_size": 2,
            "adaptive_pool_out": 64,
            "dec_kernel_sizes": [5, 5, 7, 9],
            "dec_strides": [2, 2, 4, 4],
            "dec_paddings": [2, 2, 3, 4],
            "encoder_fc": [2048, 512, 256],
            "decoder_fc": [256, 512, 2048],
            "latent_dim": 64,
            "norm_type": "group",
            "groupnorm_groups": 8,
            "batch_size": 8,
            "lr": 1e-3,
            "beta": 0.01,
            "kl_anneal_epochs": 50,
        },
    }

    if preset_name == "auto":
        if input_dim >= 75000:
            preset_name = "large"
        elif input_dim >= 10000:
            preset_name = "medium"
        else:
            preset_name = "small"

    return preset_name, presets[preset_name]


def compute_class_weights_from_data(x_long, n_classes):
    valid = (x_long >= 0) & (x_long < n_classes)
    if valid.any():
        counts = torch.bincount(x_long[valid].reshape(-1), minlength=n_classes).float()
        weights = counts.sum() / (counts + 1e-8)
        weights = weights / weights.mean()
        return weights
    return torch.ones(n_classes, dtype=torch.float32)

class Genomic_VAE_1DCNN(nn.Module):
    # Intialize the entire model with Nones
    def __init__(self, input_dim, encoder_hidden_dims, latent_dim, decoder_hidden_dims,
                 activation=nn.ReLU(), n_classes=GENOTYPES,
                 enc_channels=None, enc_kernel_sizes=None, enc_strides=None, enc_paddings=None,
                 dec_kernel_sizes=None, dec_strides=None, dec_paddings=None,
                 pool_size=4, adaptive_pool_out=16,
                 norm_type="group", groupnorm_groups=8,
                 input_encoding="onehot"):
        super(Genomic_VAE_1DCNN, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes  # Number of genotype classes (0, 1, 2)
        self.input_encoding = input_encoding
        self.input_channels = self.n_classes if self.input_encoding == "onehot" else 1

        # Encoder CNN defaults
        if enc_channels is None:
            enc_channels = [32, 64, 128]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [9] * len(enc_channels)
        if enc_strides is None:
            enc_strides = [8] + [4] * (len(enc_channels) - 1)
        if enc_paddings is None:
            enc_paddings = [4] * len(enc_channels)

        assert len(enc_channels) == len(enc_kernel_sizes) == len(enc_strides) == len(enc_paddings), \
            "Encoder CNN parameter lists must all have the same length"

        # 1D CNN Encoder - dynamically built from configuration.
        # Large strides in early layers are critical to reduce intermediate feature map
        # sizes and avoid excessive memory usage for high-dimensional input (~450K SNPs).
        encoder_layers = []
        in_ch = self.input_channels
        for i, (out_ch, ks, stride, pad) in enumerate(
            zip(enc_channels, enc_kernel_sizes, enc_strides, enc_paddings)
        ):
            encoder_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad))
            encoder_layers.append(build_norm_1d(out_ch, norm_type=norm_type, groupnorm_groups=groupnorm_groups))
            encoder_layers.append(nn.ReLU())
            if i < len(enc_channels) - 1:
                encoder_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            in_ch = out_ch
        encoder_layers.append(nn.AdaptiveAvgPool1d(adaptive_pool_out))
        self.cnn_encoder = nn.Sequential(*encoder_layers)

        self.cnn_output_dim = enc_channels[-1] * adaptive_pool_out
        self._dec_reshape_channels = enc_channels[-1]
        self._adaptive_pool_out = adaptive_pool_out

        # Fully connected layers after CNN (Encoder)
        fc_layers = []
        prev_dim = self.cnn_output_dim
        for hidden_dim in encoder_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(activation)
            prev_dim = hidden_dim
        self.fc_encoder = nn.Sequential(*fc_layers)

        # Latent space parameters
        self.mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Decoder FC stem: latent -> compact sequence embedding
        decoder_fc_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_fc_layers.append(activation)
            prev_dim = hidden_dim
        decoder_fc_layers.append(nn.Linear(prev_dim, self.cnn_output_dim))
        decoder_fc_layers.append(activation)
        self.decoder_fc = nn.Sequential(*decoder_fc_layers)

        # Decoder CNN defaults
        if dec_kernel_sizes is None:
            dec_kernel_sizes = [16] * len(enc_channels)
        if dec_strides is None:
            dec_strides = [8] * len(enc_channels)
        if dec_paddings is None:
            dec_paddings = [4] * len(enc_channels)

        assert len(dec_kernel_sizes) == len(dec_strides) == len(dec_paddings) == len(enc_channels), \
            "Decoder CNN parameter lists must match the number of encoder CNN layers"

        # Decoder CNN: progressively upsample from compact sequence → original SNP length.
        # Final Upsample bridges the remaining gap to input_dim.
        dec_in_channels = list(reversed(enc_channels))  # e.g., [128, 64, 32]
        dec_out_channels = dec_in_channels[1:] + [max(dec_in_channels[-1] // 2, 1)]  # e.g., [64, 32, 16]

        decoder_layers = []
        for in_ch, out_ch, ks, stride, pad in zip(
            dec_in_channels, dec_out_channels, dec_kernel_sizes, dec_strides, dec_paddings
        ):
            decoder_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad))
            decoder_layers.append(build_norm_1d(out_ch, norm_type=norm_type, groupnorm_groups=groupnorm_groups))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Upsample(size=self.input_dim, mode='linear', align_corners=False))
        decoder_layers.append(nn.Conv1d(dec_out_channels[-1], self.n_classes, kernel_size=1))
        self.decoder_cnn = nn.Sequential(*decoder_layers)

        # Per-SNP learnable bias: allows the model to learn position-specific
        # genotype distributions directly, bypassing the CNN upsample bottleneck.
        # Without this, the CNN decoder can only produce ~4K independent spatial
        # positions for 119K SNPs, severely limiting reconstruction capacity.
        self.snp_bias = nn.Parameter(torch.zeros(n_classes, input_dim))

    def init_snp_bias_from_data(self, x):
        """Initialize snp_bias to empirical log-frequencies from training data.
        Gives the decoder a strong starting point so the CNN pathway can focus
        on learning sample-specific deviations from the marginal frequencies."""
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.long)
            x_long = x.long()
            for c in range(self.n_classes):
                freq = (x_long == c).float().mean(dim=0)
                self.snp_bias.data[c] = torch.log(freq.clamp(min=1e-6))
        print(f"[INFO] init_snp_bias_from_data(): initialized bias from {x_long.shape[0]} samples", flush=True)

    def decode_logits(self, z):
        decoded_fc = self.decoder_fc(z)  # (batch, cnn_output_dim)
        decoded_seq = decoded_fc.view(z.size(0), self._dec_reshape_channels, self._adaptive_pool_out)
        logits = self.decoder_cnn(decoded_seq)  # (batch, n_classes, input_dim)
        logits = logits + self.snp_bias.unsqueeze(0)  # add per-SNP bias
        return logits

    def _prepare_encoder_input(self, x):
        if x.dim() == 3:
            return x.float()
        if x.dim() != 2:
            raise ValueError(f"Expected input dim 2 or 3, got {x.dim()}")

        if self.input_encoding == "onehot":
            x_long = x.long()
            valid = ((x_long >= 0) & (x_long < self.n_classes)).unsqueeze(-1)
            x_clamped = x_long.clamp(min=0, max=self.n_classes - 1)
            one_hot = F.one_hot(x_clamped, num_classes=self.n_classes).float()
            one_hot = one_hot * valid.float()
            return one_hot.permute(0, 2, 1).contiguous()

        return x.float().unsqueeze(1)

    def encode(self, x):
        x_prepared = self._prepare_encoder_input(x)
        encoded_cnn = self.cnn_encoder(x_prepared)
        encoded_flat = encoded_cnn.view(x_prepared.size(0), -1)
        encoded = self.fc_encoder(encoded_flat)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from N(mu, var) in a differentiable way.

        Args:
            mu (Tensor): shape (batch, latent_dim) - mean of the approximate posterior.
            logvar (Tensor): shape (batch, latent_dim) - log variance of the approximate posterior.

        Returns:
            Tensor: sampled latent `z` with shape (batch, latent_dim), computed as
            z = mu + sigma * eps where sigma = exp(0.5 * logvar) and eps ~ N(0, I).
        """
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_data(self, N, temperature=1.0):
        """Generate N samples by sampling from N(0, I) and decoding to class logits.
        Returns integer class predictions (0, 1, 2) for each SNP.
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            z = torch.randn(N, self.mu.out_features, device=device)
            logits = self.decode_logits(z)  # (N, n_classes, input_dim)
            # Permute to (N, input_dim, n_classes) so softmax is over genotype classes
            logits = logits.permute(0, 2, 1)  # (N, input_dim, n_classes)
            logits = logits / max(float(temperature), 1e-6)
            probs = torch.softmax(logits, dim=-1)  # softmax over n_classes
            # Sample from categorical distribution for each SNP position
            generated_data = torch.multinomial(probs.view(-1, self.n_classes), num_samples=1).view(N, self.input_dim)
        return generated_data.cpu().numpy()

    def generate_from_posterior(self, x, noise_scale=0.25, temperature=1.0):
        """Generate from posterior neighborhoods around real inputs for better manifold adherence."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = x.to(device)
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + noise_scale * eps * std
            logits = self.decode_logits(z).permute(0, 2, 1)
            logits = logits / max(float(temperature), 1e-6)
            probs = torch.softmax(logits, dim=-1)
            generated_data = torch.multinomial(probs.view(-1, self.n_classes), num_samples=1).view(x.size(0), self.input_dim)
        return generated_data.cpu().numpy()

def VariationalLoss(logits, x, mu, logvar, beta=1.0, class_weights=None,
                    freq_loss_weight=0.0, free_bits=0.0):
    """Compute VAE loss with categorical cross-entropy on decoder CNN output.

    Args:
        logits (Tensor): shape (batch, n_classes, input_dim) - raw decoder CNN output
        x (Tensor): shape (batch, input_dim) - target integer classes (0, 1, 2)
        mu (Tensor): encoder mean, shape (batch, latent_dim)
        logvar (Tensor): encoder log-variance, shape (batch, latent_dim)
        beta (float): weight for KL divergence term (default 1.0)
        free_bits (float): minimum KL per latent dim in nats (prevents posterior collapse)
    """
    # ! Reconstruction loss (Cross-entropy for categorical data)
    recon_loss = nn.CrossEntropyLoss(
        reduction='mean',
        ignore_index=-1,
        weight=class_weights,
    )(logits, x.long())

    # ! KL divergence loss with free-bits (per-dimension minimum)
    # KL per latent dimension: shape (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        # Clamp each latent dimension to a minimum of `free_bits` nats.
        # This prevents posterior collapse by ensuring the encoder must
        # encode at least `free_bits` of information per dimension.
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

    # Optional genotype-frequency alignment term (helps PCA manifold overlap)
    freq_loss = torch.tensor(0.0, device=logits.device)
    if freq_loss_weight > 0:
        n_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)  # (batch, classes, snps)
        x_long = x.long()
        valid = ((x_long >= 0) & (x_long < n_classes)).unsqueeze(1).float()
        x_clamped = x_long.clamp(min=0, max=n_classes - 1)
        target_onehot = F.one_hot(x_clamped, num_classes=n_classes).permute(0, 2, 1).float()

        denom = valid.sum(dim=(0, 2)).clamp_min(1.0)
        pred_freq = (probs * valid).sum(dim=(0, 2)) / denom
        true_freq = (target_onehot * valid).sum(dim=(0, 2)) / denom
        freq_loss = torch.mean((pred_freq - true_freq) ** 2)

    total_loss = recon_loss + beta * kl_loss + freq_loss_weight * freq_loss
    components = {
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'freq': freq_loss.item() if torch.is_tensor(freq_loss) else 0.0,
    }
    return total_loss, components

def TrainModel(model, input_data, num_epochs=200, batch_size=8, lr=0.001,
               beta=1.0, kl_anneal_epochs=0, class_weights=None,
               freq_loss_weight=0.0, free_bits=0.0, warmup_epochs=0,
               bias_lr_scale=0.1, device=None):
    # Convert input to integer tensors (categorical class labels: 0, 1, 2)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] TrainModel(): moving model to device {device}", flush=True)
    model = model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("[INFO] TrainModel(): building tensors from input data", flush=True)
    X_train_tensor = torch.tensor(input_data, dtype=torch.long)
    y_train_tensor = torch.tensor(input_data, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    # batch_size=8 is recommended for ~450K features to avoid OOM errors.
    # Increase if you have sufficient RAM/VRAM.
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )
    print(f"[INFO] TrainModel(): DataLoader ready | samples={len(ds)}, batch_size={batch_size}, batches={len(dl)}", flush=True)

    # Separate param groups: bias gets a lower LR to prevent it from dominating.
    bias_params = [p for n, p in model.named_parameters() if 'snp_bias' in n]
    other_params = [p for n, p in model.named_parameters() if 'snp_bias' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': bias_params, 'lr': lr * bias_lr_scale},
    ])
    print(f"[INFO] TrainModel(): optimizer initialized (Adam, main_lr={lr}, bias_lr={lr * bias_lr_scale})", flush=True)
    if warmup_epochs > 0:
        print(f"[INFO] TrainModel(): Phase 1 — warmup for {warmup_epochs} epochs (bias frozen, beta=0, force z-usage)", flush=True)
    loss_fx = VariationalLoss

    if class_weights is not None:
        class_weights = class_weights.to(device)

    training_loss=[]
    checked_device_once = False
    total_epochs = warmup_epochs + num_epochs
    print(f"[INFO] TrainModel(): starting training for {total_epochs} epochs "
          f"({warmup_epochs} warmup + {num_epochs} main)", flush=True)

    # ── Phase control ───────────────────────────────────────────────────
    # Phase 1 (warmup): Freeze snp_bias, set beta=0. Forces the CNN+FC+latent
    #   pathway to learn sample-specific reconstruction through z.
    # Phase 2 (main): Unfreeze snp_bias (low LR), anneal beta. Bias fine-tunes
    #   marginal frequencies while z retains sample-specific info.
    if warmup_epochs > 0:
        model.snp_bias.requires_grad_(False)

    for epoch in range(total_epochs):
        # Phase transition: unfreeze bias at start of Phase 2
        if epoch == warmup_epochs and warmup_epochs > 0:
            model.snp_bias.requires_grad_(True)
            print(f"[INFO] TrainModel(): Phase 2 — bias unfrozen, KL annealing begins", flush=True)

        in_warmup = epoch < warmup_epochs
        main_epoch = max(0, epoch - warmup_epochs)  # epoch count within Phase 2

        # Beta schedule: 0 during warmup, then linear ramp
        if in_warmup:
            current_beta = 0.0
        elif kl_anneal_epochs and kl_anneal_epochs > 0:
            current_beta = beta * min(1.0, float(main_epoch + 1) / float(kl_anneal_epochs))
        else:
            current_beta = beta

        model.train()
        total_loss = 0
        epoch_recon = 0.0
        epoch_kl = 0.0
        for batch_x, batch_y in dl:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            if not checked_device_once:
                model_device = next(model.parameters()).device
                assert model_device.type == device.type
                print(f"[INFO] Device check passed: model/data/tensors are on {model_device}", flush=True)
                checked_device_once = True

            optimizer.zero_grad()
            logits, mu, logvar = model(batch_x)
            loss, comps = loss_fx(
                logits,
                batch_y,
                mu,
                logvar,
                beta=current_beta,
                class_weights=class_weights,
                freq_loss_weight=freq_loss_weight,
                free_bits=free_bits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            epoch_recon += comps['recon']
            epoch_kl += comps['kl']
        n_batches = len(dl)
        training_loss.append(total_loss / n_batches)
        phase_tag = "WRM" if in_warmup else "MAIN"
        print(f"Epoch {epoch+1}/{total_epochs} [{phase_tag}], Loss: {total_loss/n_batches:.6f}, "
              f"Recon: {epoch_recon/n_batches:.6f}, KL: {epoch_kl/n_batches:.4f}, "
              f"beta: {current_beta:.4f}", flush=True)

    print("[INFO] TrainModel(): training complete", flush=True)

    # Plot the training loss
    plt.plot(training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

def parse_int_list(s):
    """Parse a comma-separated string of ints, e.g. '32,64,128' -> [32,64,128]."""
    return [int(x.strip()) for x in s.split(',')]


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train a 1D-CNN Variational Autoencoder on genomic (SNP) data "
                    "and generate synthetic samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset I/O ──────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to input CSV dataset (rows=samples, cols=SNPs).')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save generated CSV. Defaults to '
                             '<dataset_dir>/<dataset_name>_generated.csv.')
    parser.add_argument('--snp-subset', type=str, default=None,
                        help='Optional text file with SNP column names to subset.')

    # ── Encoder CNN ─────────────────────────────────────────────────────
    parser.add_argument('--arch-preset', type=str, choices=['auto', 'small', 'medium', 'large'],
                        default='auto',
                        help='Feature-count aware architecture preset. `auto` selects based on input dimensionality.')
    parser.add_argument('--enc-channels', type=parse_int_list, default=None,
                        help='Comma-separated encoder CNN channel sizes. Defaults come from `--arch-preset`.')
    parser.add_argument('--enc-kernel-sizes', type=parse_int_list, default=None,
                        help='Comma-separated encoder kernel sizes (one per layer). '
                            'Defaults come from `--arch-preset`.')
    parser.add_argument('--enc-strides', type=parse_int_list, default=None,
                        help='Comma-separated encoder strides. '
                            'Defaults come from `--arch-preset`.')
    parser.add_argument('--enc-paddings', type=parse_int_list, default=None,
                        help='Comma-separated encoder paddings. Defaults come from `--arch-preset`.')
    parser.add_argument('--pool-size', type=int, default=None,
                        help='MaxPool1d kernel/stride between encoder CNN layers. Defaults come from `--arch-preset`.')
    parser.add_argument('--adaptive-pool-out', type=int, default=None,
                        help='Output size of AdaptiveAvgPool1d in encoder. Defaults come from `--arch-preset`.')
    parser.add_argument('--norm-type', type=str, choices=['group', 'batch', 'layer', 'none'],
                        default=None,
                        help='Normalization type in encoder/decoder CNN blocks. '
                            'GroupNorm is usually more stable than BatchNorm for small batch sizes. '
                            'Defaults come from `--arch-preset`.')
    parser.add_argument('--groupnorm-groups', type=int, default=None,
                        help='Number of groups for GroupNorm when `--norm-type group` is used. Defaults come from `--arch-preset`.')

    # ── Decoder CNN ─────────────────────────────────────────────────────
    parser.add_argument('--dec-kernel-sizes', type=parse_int_list, default=None,
                        help='Comma-separated decoder kernel sizes. Defaults come from `--arch-preset`.')
    parser.add_argument('--dec-strides', type=parse_int_list, default=None,
                        help='Comma-separated decoder strides. Defaults come from `--arch-preset`.')
    parser.add_argument('--dec-paddings', type=parse_int_list, default=None,
                        help='Comma-separated decoder paddings. Defaults come from `--arch-preset`.')

    # ── FC / Latent ─────────────────────────────────────────────────────
    parser.add_argument('--encoder-fc', type=parse_int_list,
                        default=None,
                        help='Comma-separated encoder FC hidden dims. Defaults come from `--arch-preset`.')
    parser.add_argument('--decoder-fc', type=parse_int_list,
                        default=None,
                        help='Comma-separated decoder FC hidden dims. Defaults come from `--arch-preset`.')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Dimensionality of the latent space. Defaults come from `--arch-preset`.')
    parser.add_argument('--n-classes', type=int, default=3,
                        help='Number of genotype classes (default 3 for 0/1/2).')
    parser.add_argument('--input-encoding', type=str, choices=['onehot', 'dosage'],
                        default='onehot',
                        help='Input encoding to the encoder. `onehot` is usually better for categorical genotype states.')

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size. Defaults come from `--arch-preset`.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for Adam optimizer. Defaults come from `--arch-preset`.')
    parser.add_argument('--beta', type=float, default=None,
                        help='Beta weight for KL divergence in the loss. Defaults come from `--arch-preset`.')
    parser.add_argument('--kl-anneal-epochs', type=int, default=None,
                        help='Linearly ramp beta from 0 to `--beta` over this many epochs. '
                             'Useful when generated samples sit far from the real data distribution.')
    parser.add_argument('--freq-loss-weight', type=float, default=0.05,
                        help='Weight for genotype-frequency matching loss term. Set 0 to disable.')
    parser.add_argument('--free-bits', type=float, default=2.0,
                        help='Minimum KL per latent dimension in nats. Prevents posterior collapse '
                             'by forcing the encoder to encode at least this much info per dim.')
    parser.add_argument('--warmup-epochs', type=int, default=50,
                        help='Phase 1 warmup epochs: bias frozen, beta=0, forces z-based '
                             'reconstruction before introducing KL penalty.')
    parser.add_argument('--bias-lr-scale', type=float, default=0.1,
                        help='Learning rate multiplier for snp_bias relative to main LR. '
                             'Lower values prevent bias from dominating reconstruction.')
    parser.add_argument('--disable-class-weights', action='store_true',
                        help='Disable inverse-frequency class weighting in reconstruction CE loss.')

    # ── Generation ──────────────────────────────────────────────────────
    parser.add_argument('--n-generate', type=int, default=100,
                        help='Number of synthetic samples to generate after training.')
    parser.add_argument('--generation-mode', type=str, choices=['prior', 'posterior', 'hybrid'],
                        default='hybrid',
                        help='Generation source: prior, posterior neighborhoods, or half/half hybrid.')
    parser.add_argument('--posterior-noise-scale', type=float, default=0.25,
                        help='Noise scale for posterior sampling in `posterior`/`hybrid` generation modes.')
    parser.add_argument('--sample-temperature', type=float, default=1.0,
                        help='Sampling temperature for genotype categorical draws (lower is sharper).')

    # ── SLURM Integration ────────────────────────────────────────
    parser.add_argument('--slurm-id', type=str, default=None,
                        help='SLURM job ID to include in output filename for tracking.')

    return parser


def main():
    args = build_argparser().parse_args()

    # ── Load dataset ────────────────────────────────────────────────────
    print(f"[INFO] main(): loading dataset from {args.dataset}", flush=True)
    input_data = pd.read_csv(args.dataset)
    print(f"[INFO] main(): dataset loaded with shape {input_data.shape}", flush=True)

    if args.snp_subset is not None:
        sampled_cols = np.loadtxt(args.snp_subset, dtype=str)
        input_data = input_data[sampled_cols]
        print(f"[INFO] main(): subset to {input_data.shape[1]} SNPs from {args.snp_subset}", flush=True)

    values = input_data.values
    columns = input_data.columns

    # ── Build model ─────────────────────────────────────────────────────
    print("[INFO] main(): initializing model", flush=True)
    input_dim = input_data.shape[1]
    preset_name, preset = get_architecture_preset(input_dim, args.arch_preset)
    resolved_config = {
        'enc_channels': pick_override(args.enc_channels, preset['enc_channels']),
        'enc_kernel_sizes': pick_override(args.enc_kernel_sizes, preset['enc_kernel_sizes']),
        'enc_strides': pick_override(args.enc_strides, preset['enc_strides']),
        'enc_paddings': pick_override(args.enc_paddings, preset['enc_paddings']),
        'pool_size': pick_override(args.pool_size, preset['pool_size']),
        'adaptive_pool_out': pick_override(args.adaptive_pool_out, preset['adaptive_pool_out']),
        'dec_kernel_sizes': pick_override(args.dec_kernel_sizes, preset['dec_kernel_sizes']),
        'dec_strides': pick_override(args.dec_strides, preset['dec_strides']),
        'dec_paddings': pick_override(args.dec_paddings, preset['dec_paddings']),
        'encoder_fc': pick_override(args.encoder_fc, preset['encoder_fc']),
        'decoder_fc': pick_override(args.decoder_fc, preset['decoder_fc']),
        'latent_dim': pick_override(args.latent_dim, preset['latent_dim']),
        'norm_type': pick_override(args.norm_type, preset['norm_type']),
        'groupnorm_groups': pick_override(args.groupnorm_groups, preset['groupnorm_groups']),
        'batch_size': pick_override(args.batch_size, preset['batch_size']),
        'lr': pick_override(args.lr, preset['lr']),
        'beta': pick_override(args.beta, preset['beta']),
        'kl_anneal_epochs': pick_override(args.kl_anneal_epochs, preset['kl_anneal_epochs']),
    }
    print(f"[INFO] main(): using architecture preset '{preset_name}' for input_dim={input_dim}", flush=True)
    print(
        "[INFO] main(): resolved config | "
        f"enc_channels={resolved_config['enc_channels']}, "
        f"enc_strides={resolved_config['enc_strides']}, "
        f"pool_size={resolved_config['pool_size']}, "
        f"adaptive_pool_out={resolved_config['adaptive_pool_out']}, "
        f"encoder_fc={resolved_config['encoder_fc']}, "
        f"latent_dim={resolved_config['latent_dim']}, "
        f"batch_size={resolved_config['batch_size']}, "
        f"lr={resolved_config['lr']}, "
        f"beta={resolved_config['beta']}, "
        f"kl_anneal_epochs={resolved_config['kl_anneal_epochs']}",
        flush=True,
    )
    model = Genomic_VAE_1DCNN(
        input_dim=input_dim,
        encoder_hidden_dims=resolved_config['encoder_fc'],
        latent_dim=resolved_config['latent_dim'],
        decoder_hidden_dims=resolved_config['decoder_fc'],
        n_classes=args.n_classes,
        enc_channels=resolved_config['enc_channels'],
        enc_kernel_sizes=resolved_config['enc_kernel_sizes'],
        enc_strides=resolved_config['enc_strides'],
        enc_paddings=resolved_config['enc_paddings'],
        dec_kernel_sizes=resolved_config['dec_kernel_sizes'],
        dec_strides=resolved_config['dec_strides'],
        dec_paddings=resolved_config['dec_paddings'],
        pool_size=resolved_config['pool_size'],
        adaptive_pool_out=resolved_config['adaptive_pool_out'],
        norm_type=resolved_config['norm_type'],
        groupnorm_groups=resolved_config['groupnorm_groups'],
        input_encoding=args.input_encoding,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[INFO] main(): using device {device}", flush=True)
    print(f"[INFO] main(): model parameters are on {next(model.parameters()).device}", flush=True)

    class_weights = None
    if not args.disable_class_weights:
        class_weights = compute_class_weights_from_data(torch.tensor(values, dtype=torch.long), args.n_classes)
        print(f"[INFO] main(): class weights={class_weights.tolist()}", flush=True)

    # ── Initialize per-SNP bias from empirical frequencies ───────────
    print("[INFO] main(): initializing per-SNP bias from training data frequencies", flush=True)
    model.init_snp_bias_from_data(torch.tensor(values, dtype=torch.long))

    # ── Train ───────────────────────────────────────────────────────────
    print("[INFO] main(): training model", flush=True)
    TrainModel(model, values, num_epochs=args.epochs, batch_size=resolved_config['batch_size'],
               lr=resolved_config['lr'], beta=resolved_config['beta'],
               kl_anneal_epochs=resolved_config['kl_anneal_epochs'],
               class_weights=class_weights,
               freq_loss_weight=args.freq_loss_weight,
               free_bits=args.free_bits,
               warmup_epochs=args.warmup_epochs,
               bias_lr_scale=args.bias_lr_scale,
               device=device)

    # ── Generate ────────────────────────────────────────────────────────
    model.eval()
    print(f"[INFO] main(): generating {args.n_generate} synthetic samples", flush=True)
    if args.generation_mode == 'prior':
        data = model.generate_data(args.n_generate, temperature=args.sample_temperature)
    elif args.generation_mode == 'posterior':
        sampled_idx = np.random.choice(values.shape[0], size=args.n_generate, replace=True)
        seed_x = torch.tensor(values[sampled_idx], dtype=torch.long)
        data = model.generate_from_posterior(
            seed_x,
            noise_scale=args.posterior_noise_scale,
            temperature=args.sample_temperature,
        )
    else:
        n_post = args.n_generate // 2
        n_prior = args.n_generate - n_post
        sampled_idx = np.random.choice(values.shape[0], size=n_post, replace=True)
        seed_x = torch.tensor(values[sampled_idx], dtype=torch.long)
        post_data = model.generate_from_posterior(
            seed_x,
            noise_scale=args.posterior_noise_scale,
            temperature=args.sample_temperature,
        )
        prior_data = model.generate_data(n_prior, temperature=args.sample_temperature)
        data = np.vstack([post_data, prior_data])

    # ── Save ────────────────────────────────────────────────────────────
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.dataset))[0]
        if args.slurm_id:
            output_filename = f"{base}_generated_slurm_{args.slurm_id}.csv"
        else:
            output_filename = f"{base}_generated.csv"
        # Use absolute results/gen_data under project root
        output_path = os.path.join(os.getcwd(), "results", "gen_data", output_filename)
    else:
        output_path = args.output

    print(f"[INFO] main(): saving generated data to {output_path}", flush=True)
    generated_df = pd.DataFrame(data, columns=columns)
    # Ensure parent directory exists before saving
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    generated_df.to_csv(output_path, index=False)
    print("[INFO] main(): finished", flush=True)


if __name__ == "__main__":
    main()