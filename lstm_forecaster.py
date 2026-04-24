import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader

class LSTMForecaster(nn.Module):
    """
    Standard downstream demand forecasting model.
    Takes 21 days of historical demand and predicts the next 7 days.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=7):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out) # Demand is normalized between 0 and 1

def train_lstm():
    # 1. Load Data (We only want to train on "Normal" data!)
    dataloader = get_dataloader(batch_size=64, num_samples=5000, sequence_length=28)
    
    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Downstream LSTM Forecaster on Normal Data...")
    
    epochs = 40
    for epoch in range(epochs):
        train_loss = 0
        batches = 0
        for batch_data, batch_cond in dataloader:
            # Filter for normal conditions only (condition == 0)
            normal_mask = (batch_cond == 0)
            if not normal_mask.any():
                continue
                
            normal_data = batch_data[normal_mask]
            
            # Input: first 21 days. Target: last 7 days.
            # Add an extra dimension for feature size = 1
            x = normal_data[:, :21].unsqueeze(2) 
            y = normal_data[:, 21:]
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss/batches:.4f}')
            
    print("LSTM Training Complete!")
    return model

if __name__ == "__main__":
    trained_lstm = train_lstm()
    # Save the model so we can stress test it in evaluate.py
    torch.save(trained_lstm.state_dict(), "lstm_model.pth")
    print("Saved LSTM model to 'lstm_model.pth'")
