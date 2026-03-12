from numpy.random import beta
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale

class DataGenerator(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, latent_dim, decoder_hidden_dims, activation=nn.ReLU()):
        super(DataGenerator, self).__init__()

        # Append the encoder layers
        encoder_layers=[]
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation)
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Append the decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(activation)
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_data(self, N):
        z = torch.randn(N, self.mu.out_features)
        generated_data = self.decoder(z)
        return generated_data

def GetInputData(n=1000):
    lifestyle = np.random.normal(0,1,n)

    income_mean = 50000.0
    income_std = 15000.0
    income = np.random.normal(loc=income_mean, scale=income_std, size=n)

    house_size_mean = 1500.0
    house_size_std = 300.0
    house_size = np.random.normal(loc=house_size_mean, scale=house_size_std, size=n)

    min_people, max_people = 1, 6
    house_norm = (house_size - house_size.min()) / (house_size.max() - house_size.min())
    num_people = np.round(np.clip(min_people + house_norm * (max_people - min_people), min_people, max_people))

    temp_mean = 4.225
    temp_std = 0.212
    outdoor_temp = np.random.lognormal(mean=temp_mean, sigma=temp_std, size=n)

    hvac = outdoor_temp * 0.5 + np.random.randn(n)
    appliance = lifestyle*3 + np.random.randn(n)
    solar = 0.10*outdoor_temp + np.random.randn(n)

    total_energy = hvac + appliance - solar

    df = pd.DataFrame({
        "income":income,
        "house_size":house_size,
        "num_people":num_people,
        "outdoor_temp":outdoor_temp,
        "hvac_usage":hvac,
        "appliance_usage":appliance,
        "solar_generation":solar,
        "total_energy":total_energy
    })

    df.to_csv("input_data.csv", index=False)
    return df

def VariationalLoss(recon_x, x, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Add stddev loss to encourage diversity in the generated data
    std_loss = torch.mean((x.std(dim=0) - recon_x.std(dim=0))**2)

    beta = 1.0
    gamma = 1.0
    total_loss = recon_loss + beta * kl_loss + gamma * std_loss
    return total_loss

def TrainModel(model, input_data, num_epochs=100):
    X_train_tensor = torch.tensor(input_data, dtype=torch.float32)
    y_train_tensor = torch.tensor(input_data, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fx = VariationalLoss

    training_loss=[]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dl:
            optimizer.zero_grad()
            recon_x, mu, log_var = model(batch_x)
            loss = loss_fx(recon_x, batch_y, mu, log_var)
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

if __name__ == "__main__":
    # Create an instance of the DataGenerator
    model = DataGenerator(input_dim=8, encoder_hidden_dims=[16, 16], latent_dim=2, decoder_hidden_dims=[32, 16])
    
    # Generate input data
    input_data = GetInputData(n=1000)

    # Scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Train the model
    TrainModel(model, input_data_scaled, num_epochs=100)
    
    # Generate a batch of data
    batch_size = 256
    data = model.generate_data(batch_size)

    # Inverse transform the generated data to the original scale
    data_original_scale = scaler.inverse_transform(data.detach().numpy())

    # Round the generated data to appropriate values (e.g., num_people should be an integer)
    data_original_scale[:, 2] = np.round(data_original_scale[:, 2])  # Round num_people to nearest integer
    
    # Save the generated data to a CSV file
    generated_df = pd.DataFrame(data_original_scale, columns=["income", "house_size", "num_people", "outdoor_temp", "hvac_usage", "appliance_usage", "solar_generation", "total_energy"])
    generated_df.to_csv("generated_data.csv", index=False)