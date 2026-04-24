import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Base Variational Autoencoder (VAE) for time-series demand data.
    """
    def __init__(self, input_dim=28, hidden_dim=16, latent_dim=4):
        super(VAE, self).__init__()
        
        # --- ENCODER ---
        # The encoder compresses the input into a lower-dimensional latent space.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Instead of outputting a single vector, the VAE encoder outputs two vectors:
        # 1. mu: The mean of the latent distribution
        # 2. log_var: The log-variance of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- DECODER ---
        # The decoder takes a point from the latent space and reconstructs the input.
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Passes input through the encoder to get mean and log-variance."""
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_log_var(h1)

    def reparameterize(self, mu, log_var):
        """
        The Reparameterization Trick.
        Allows backpropagation through the random sampling process.
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * log_var) # Convert log-variance to standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from standard normal distribution
        return mu + eps * std

    def decode(self, z):
        """Passes latent vector through decoder to reconstruct input."""
        h3 = F.relu(self.fc3(z))
        # We use sigmoid because our synthetic data is normalized between 0 and 1
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """Full forward pass: Encode -> Sample -> Decode"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

def vae_loss(reconstructed_x, x, mu, log_var):
    """
    The loss function for a VAE consists of two parts:
    1. Reconstruction Loss: How well the decoder recreates the input.
    2. KL Divergence: How closely the latent distribution matches a standard normal distribution.
    """
    # 1. Reconstruction Loss (Binary Cross Entropy since data is 0-1 normalized)
    # Using reduction='sum' to match standard VAE implementations
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # 2. KL Divergence
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_divergence, recon_loss, kl_divergence
