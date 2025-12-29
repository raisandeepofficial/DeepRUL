import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from src.models import LSTMAttention
from src.data_loader import get_data_loaders

# load config
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def train():
    print("initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # data
    print("loading data...")
    train_loader, _ = get_data_loaders()
    
    # model
    model = LSTMAttention(cfg['input_dim'], cfg['hidden_dim'], cfg['layer_dim'], cfg['output_dim'])
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # loop
    print(f"starting training for {cfg['num_epochs']} epochs")
    for epoch in range(cfg['num_epochs']):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f'epoch {epoch+1}/{cfg["num_epochs"]}, loss: {avg_loss:.4f}')
            
    # save
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print("training complete. model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    train()
