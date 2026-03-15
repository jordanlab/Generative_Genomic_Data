from numpy.random import beta
import torch.nn as nn
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale

# Try running on a small subset of SNPs first to test the model and then scale up to the full dataset.
# Also try to see if SNP loadings could be used to identify which SNPs are most important for the latent space representation and then focus on those SNPs for training the model. 
# This could help reduce the dimensionality of the input data and make training more efficient.

GENOTYPES = 3

class Genomic_VAE_1DCNN(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, latent_dim, decoder_hidden_dims, activation=nn.ReLU(), n_classes=GENOTYPES):
        super(Genomic_VAE_1DCNN, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes  # Number of genotype classes (0, 1, 2)

        # 1D CNN Encoder for efficient feature extraction from long sequences.
        # Designed for very high-dimensional input (~450K SNP features).
        # Large strides in early layers are critical to reduce intermediate feature map
        # sizes and avoid excessive memory usage.
        # For input_dim = 450,000:
        #   Layer 1 Conv (stride=8):  450,000 → 56,250   | mem ~58 MB/batch-of-8
        #   MaxPool (stride=4):        56,250 → 14,062
        #   Layer 2 Conv (stride=4):  14,062 →  3,516
        #   MaxPool (stride=4):        3,516 →    879
        #   Layer 3 Conv (stride=4):    879 →    220
        #   AdaptiveAvgPool → 16                          | fixed 2048 output
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=8, padding=4),   # ~450K → ~56,250
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),                   # ~56,250 → ~14,062
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),  # ~14,062 → ~3,516
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),                   # ~3,516 → ~879
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4), # ~879 → ~220
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),                                # → (batch, 128, 16) fixed
        )
        # Fixed CNN output dimension regardless of input_dim (due to AdaptiveAvgPool1d)
        self.cnn_output_dim = 128 * 16  # 2048

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

        # Decoder CNN head: progressively upsample from compact sequence → original SNP length.
        # ConvTranspose1d(kernel=16, stride=8, padding=4) gives exact 8x upsampling:
        #   output = (L_in - 1) * 8 - 8 + 16 = L_in * 8
        #   16 → 128 → 1024 → 8192 → Upsample(input_dim)
        # Final Upsample bridges ~8K → 450K (factor ~55x), which is far more reasonable
        # than the previous single jump of 128 → 450K (factor ~3,500x).
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4),  # 16 → 128
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=8, padding=4),   # 128 → 1,024
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=8, padding=4),   # 1,024 → 8,192
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(size=self.input_dim, mode='linear', align_corners=False),  # 8,192 → input_dim
            nn.Conv1d(16, self.n_classes, kernel_size=1),                           # channel projection → n_classes
        )

    def decode_logits(self, z):
        decoded_fc = self.decoder_fc(z)  # (batch, 2048)
        decoded_seq = decoded_fc.view(z.size(0), 128, 16)  # (batch, 128, 16)
        logits_cnn = self.decoder_cnn(decoded_seq)  # (batch, n_classes, input_dim)
        logits = logits_cnn.permute(0, 2, 1).contiguous()  # (batch, input_dim, n_classes)
        return logits

    def forward(self, x):
        # x: (batch, input_dim)
        # Reshape for CNN: (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded_cnn = self.cnn_encoder(x)  # (batch, 128, 16)
        encoded_flat = encoded_cnn.view(x.size(0), -1)  # (batch, 2048)
        encoded = self.fc_encoder(encoded_flat)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

    # ? Not sure what this does
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

    # ? Not sure what this does
    def generate_data(self, N):
        """Generate N samples by sampling from N(0, I) and decoding to class logits.
        Returns integer class predictions (0, 1, 2) for each SNP.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(N, self.mu.out_features)
            logits = self.decode_logits(z)
            # Sample from softmax distribution or take argmax
            probs = torch.softmax(logits, dim=-1)  # shape: (N, input_dim, 3)
            # Sample from categorical distribution
            generated_data = torch.multinomial(probs.view(-1, self.n_classes), num_samples=1).view(N, self.input_dim)
        return generated_data.cpu().numpy()

def VariationalLoss(logits, x, mu, logvar):
    """Compute VAE loss with categorical cross-entropy for genotype prediction.
    
    Args:
        logits (Tensor): shape (batch, input_dim, n_classes) - decoder output logits
        x (Tensor): shape (batch, input_dim) - target integer classes (0, 1, 2)
        mu (Tensor): encoder mean
        logvar (Tensor): encoder log-variance
    """
    # ! Reconstruction loss (Cross-entropy for categorical data)
    # Flatten logits and targets for cross-entropy
    batch_size = logits.shape[0]
    input_dim = logits.shape[1]
    logits_flat = logits.view(-1, logits.shape[2])  # (batch*input_dim, n_classes)
    x_flat = x.long().view(-1)  # (batch*input_dim,)
    recon_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(logits_flat, x_flat)
    
    # ! KL divergence loss
    # Mu = the mean of the latent space distribution, which is learned by the encoder
    # logvar = the log variance of the latent space distribution, which is also learned by the encoder
    # KL Divergence between latent space distribution (assumed to be N(0,1))and the real input probability distribution
    std_norm_mean = 0
    std_norm_var = 1.0
    kl_loss = -0.5 * torch.sum(logvar + (std_norm_mean - mu.pow(2)) + (std_norm_var - logvar.exp())) 

    beta = 1.0
    total_loss = recon_loss + beta * kl_loss
    return total_loss

def TrainModel(model, input_data, num_epochs=200, batch_size=8):
    # Convert input to integer tensors (categorical class labels: 0, 1, 2)
    X_train_tensor = torch.tensor(input_data, dtype=torch.long)
    y_train_tensor = torch.tensor(input_data, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    # batch_size=8 is recommended for ~450K features to avoid OOM errors.
    # Increase if you have sufficient RAM/VRAM.
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fx = VariationalLoss

    training_loss=[]
    for epoch in range(num_epochs):
        model.train() # put model in training mode 
        total_loss = 0 # initialize total loss for the epoch
        for batch_x, batch_y in dl:
            # Convert batch to float for forward pass (encoder expects float input)
            batch_x_float = batch_x.float()
            optimizer.zero_grad() # reset gradients for the batch
            logits, mu, logvar = model(batch_x_float) # forward pass through the model to get logits and latent parameters
            loss = loss_fx(logits, batch_y, mu, logvar)  # batch_y remains long for cross-entropy
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        training_loss.append(total_loss/len(dl))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dl)}")

    # Plot the training loss
    plt.plot(training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

def main():
    # Dataframe input data
    input_data = pd.read_csv("data/BAQ/BAQ_unphased_tensor.csv")

    # For very large SNP sets (e.g., 450,000+), you may want to select a subset for initial testing,
    # or use all features if you have sufficient memory and compute.
    # sampled_cols = np.loadtxt('subset_snps.txt', dtype=str)  # Uncomment to use a subset
    # input_data = input_data[sampled_cols]

    # Save subset for PCA analysis if needed
    # input_data.to_csv("data/BAQ/BAQ_unphased_tensor_subset.csv", index=False)

    values = input_data.values  # Convert DataFrame to NumPy array
    columns = input_data.columns

    # Create an instance of the Genomic_VAE
    input_dim = input_data.shape[1]
    encoder_hidden_dims = [1024, 512, 256, 128, 64, 32, 16 , 8, 4] # Decreasing hidden layer sizes for the encoder
    latent_dim = 2  # Size of the latent space (bottleneck)
    decoder_hidden_dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Increasing hidden layer sizes for the decoder (mirroring the encoder)
    n_classes = 3
    model = Genomic_VAE_1DCNN(input_dim=input_dim, encoder_hidden_dims=encoder_hidden_dims, latent_dim=latent_dim, decoder_hidden_dims=decoder_hidden_dims, n_classes=n_classes)

    # Train the model
    TrainModel(model, values, num_epochs=100)

    # Generate a batch of data
    batch_size = 100
    data = model.generate_data(batch_size)

    # Save the generated data to a CSV file
    generated_df = pd.DataFrame(data, columns=columns)
    generated_df.to_csv("BAQ_generated_data.csv", index=False)

if __name__ == "__main__":
    main()