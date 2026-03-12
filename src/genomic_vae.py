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

class Genomic_VAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, latent_dim, decoder_hidden_dims, activation=nn.ReLU(), n_classes=3):
        super(Genomic_VAE, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes  # Number of genotype classes (0, 1, 2)

        # Encoder layers
        encoder_layers=[]
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation)
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Decoder layers (output logits for each class per SNP)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(activation)
            prev_dim = hidden_dim
        # Final layer outputs logits for all classes across all SNPs
        decoder_layers.append(nn.Linear(prev_dim, input_dim * n_classes))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)  # shape: (batch, input_dim * n_classes)
        # Reshape to (batch, input_dim, n_classes) for per-SNP softmax
        logits = logits.view(-1, self.input_dim, self.n_classes)
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
            logits = self.decoder(z)  # shape: (N, input_dim * n_classes)
            logits = logits.view(N, self.input_dim, self.n_classes)
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

def TrainModel(model, input_data, num_epochs=200):
    # Convert input to integer tensors (categorical class labels: 0, 1, 2)
    X_train_tensor = torch.tensor(input_data, dtype=torch.long)
    y_train_tensor = torch.tensor(input_data, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

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
    
    # ! Start with a small subset of the data for testing
    # Randomly choose `n_features` columns (features/SNPs)
    n_features = 250
    n_classes = 3
    seed = 42 # reproducible sampling: set a seed for numpy/torch/python random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if input_data.shape[1] > n_features:
        sampled_cols = np.random.choice(input_data.columns, size=n_features, replace=False)
        # ensure list for deterministic column ordering in downstream code
        sampled_cols = list(sampled_cols)
    else:
        sampled_cols = list(input_data.columns)
    
    sampled_cols = np.loadtxt('subset_snps.txt', dtype=str)  # Load the list of SNPs from the text file
    input_data = input_data.iloc[:][sampled_cols]
    
    # Save subset for for PCA analysis 
    input_data.to_csv("data/BAQ/BAQ_unphased_tensor_subset.csv", index=False)  # Save the subset for reference
    
    values = input_data.values  # Convert DataFrame to NumPy array
    columns = input_data.columns
    
    # Create an instance of the Genomic_VAE
    input_dim = input_data.shape[1]
    encoder_hidden_dims = [128, 64, 32, 16, 8, 4]  # ! Decreasing hidden layer sizes for the encoder
    latent_dim = 2 # ! Size of the latent space (bottleneck)
    decoder_hidden_dims = [4, 8, 16, 32, 64, 128]  # ! Increasing hidden layer sizes for the decoder (mirroring the encoder)
    model = Genomic_VAE(input_dim=input_dim, encoder_hidden_dims=encoder_hidden_dims, latent_dim=latent_dim, decoder_hidden_dims=decoder_hidden_dims, n_classes=n_classes)
    
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