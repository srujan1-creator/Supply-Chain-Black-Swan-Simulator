import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_dataloader
from vae_base import VAE, vae_loss

def train():
    # 1. Setup Data and Model
    sequence_length = 28 # Example: 4 weeks of daily data
    dataloader = get_dataloader(batch_size=32, num_samples=2000, sequence_length=sequence_length)
    
    model = VAE(input_dim=sequence_length, hidden_dim=16, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50

    model.train()
    print("Starting training...")
    
    # 2. Training Loop
    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_batch, mu, log_var = model(batch)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(reconstructed_batch, batch, mu, log_var)
            
            # Backward pass and optimization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Average Loss: {train_loss / len(dataloader.dataset):.4f}')

    print("Training complete!")
    return model

def visualize_reconstruction(model):
    """Visualizes an original demand sequence vs its VAE reconstruction."""
    model.eval()
    dl = get_dataloader(batch_size=1)
    sample = next(iter(dl))
    
    with torch.no_grad():
        reconstructed_sample, _, _ = model(sample)
    
    plt.figure(figsize=(10, 4))
    plt.plot(sample[0].numpy(), label='Original Demand', marker='o')
    plt.plot(reconstructed_sample[0].numpy(), label='Reconstructed Demand', marker='x')
    plt.title('VAE Reconstruction of Demand Data')
    plt.xlabel('Days')
    plt.ylabel('Normalized Demand')
    plt.legend()
    plt.savefig('vae_reconstruction.png')
    print("Saved reconstruction plot to 'vae_reconstruction.png'")

if __name__ == "__main__":
    trained_model = train()
    visualize_reconstruction(trained_model)
