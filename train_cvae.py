import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import get_dataloader
from cvae import CVAE, cvae_loss

def train():
    sequence_length = 28
    # Using larger num_samples to ensure we see enough of each condition
    dataloader = get_dataloader(batch_size=64, num_samples=3000, sequence_length=sequence_length)
    
    model = CVAE(input_dim=sequence_length, num_classes=3, hidden_dim=32, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 60

    model.train()
    print("Starting CVAE training...")
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_data, batch_cond in dataloader:
            optimizer.zero_grad()
            
            # Forward pass (now we pass both data and conditions)
            reconstructed_batch, mu, log_var = model(batch_data, batch_cond)
            
            # Calculate loss
            loss, recon_loss, kl_loss = cvae_loss(reconstructed_batch, batch_data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Average Loss: {train_loss / len(dataloader.dataset):.4f}')

    print("Training complete!")
    return model

def generate_conditional_samples(model):
    """
    Demonstrates the true power of the CVAE:
    We pick a random point in the latent space, and ask the decoder to decode it
    3 times, once for each condition.
    """
    model.eval()
    
    # 1. Pick a random point from the standard normal distribution (our "cloud")
    # Shape: (1 sample, latent_dim=4)
    z = torch.randn(1, 4)
    
    # 2. Define our conditions: 0 (Normal), 1 (Port Closure), 2 (Spike)
    conditions = torch.tensor([0, 1, 2])
    # Convert to one-hot encoding
    c_one_hot = F.one_hot(conditions, num_classes=3).float()
    
    # We need to duplicate z so it matches the number of conditions (3)
    z_repeated = z.repeat(3, 1)
    
    # 3. Decode!
    with torch.no_grad():
        generated_samples = model.decode(z_repeated, c_one_hot)
    
    # 4. Visualize
    plt.figure(figsize=(12, 6))
    
    labels = ["Normal", "Port Closure", "Demand Spike"]
    colors = ['blue', 'red', 'green']
    
    for i in range(3):
        plt.plot(generated_samples[i].numpy(), label=labels[i], color=colors[i], linewidth=2)
        
    plt.title('CVAE Generative Stress Testing: Same Latent Core, Different Disruption Types')
    plt.xlabel('Days')
    plt.ylabel('Normalized Demand')
    plt.legend()
    plt.savefig('cvae_generation.png')
    print("Saved generated scenario plot to 'cvae_generation.png'")

if __name__ == "__main__":
    trained_model = train()
    generate_conditional_samples(trained_model)
