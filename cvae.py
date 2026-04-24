import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for time-series demand data.
    Allows us to generate demand curves specific to a given condition
    (e.g., 0=Normal, 1=Port Closure, 2=Demand Spike).
    """
    def __init__(self, input_dim=28, num_classes=3, hidden_dim=32, latent_dim=4):
        super(CVAE, self).__init__()
        
        self.num_classes = num_classes
        
        # --- ENCODER ---
        # The encoder now takes the original data PLUS the one-hot encoded condition.
        # So the input size is input_dim + num_classes
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- DECODER ---
        # The decoder takes the latent vector z PLUS the one-hot encoded condition.
        # So the input size is latent_dim + num_classes
        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        """
        x: batch of demand sequences (batch_size, input_dim)
        c: batch of one-hot encoded conditions (batch_size, num_classes)
        """
        # Concatenate data and condition
        inputs = torch.cat([x, c], dim=1) 
        h1 = F.relu(self.fc1(inputs))
        return self.fc_mu(h1), self.fc_log_var(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """
        z: batch of latent vectors
        c: batch of one-hot encoded conditions
        """
        # Concatenate latent vector and condition
        inputs = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, condition_labels):
        """
        condition_labels: tensor of integers (e.g., 0, 1, or 2)
        """
        # Convert integer labels to one-hot encoding
        c = F.one_hot(condition_labels, num_classes=self.num_classes).float()
        
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z, c)
        
        return reconstructed_x, mu, log_var

# We can reuse the same loss function from the base VAE!
def cvae_loss(reconstructed_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence, recon_loss, kl_divergence
