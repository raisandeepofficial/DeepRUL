import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.models import LSTMAttention
from src.data_loader import get_data_loaders

def evaluate():
    # 1. load config
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. load data
    print("loading data for evaluation...")
    _, test_loader = get_data_loaders()
    
    # 3. load model
    model = LSTMAttention(cfg['input_dim'], cfg['hidden_dim'], cfg['layer_dim'], cfg['output_dim'])
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.to(device)
    model.eval()
    
    # 4. make predictions
    preds = []
    actuals = []
    
    print("running inference...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            preds.extend(output.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
            
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    # --- PLOT 1: Prediction vs Actual (The "Money Shot") ---
    print("generating plot 1: prediction vs actual...")
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:300], label='Actual RUL', color='black', linewidth=2)
    plt.plot(preds[:300], label='Predicted RUL', color='#00ffcc', linestyle='--')
    plt.title('DeepRUL: Prediction vs Actual (First 300 samples)', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('01_predictions_vs_actual.png')
    plt.close()

    # --- PLOT 2: Error Distribution (The "Safety Check") ---
    print("generating plot 2: error distribution...")
    errors = preds - actuals
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='purple')
    plt.title('Error Distribution (Residuals)', fontsize=14)
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig('02_error_distribution.png')
    plt.close()
    
    # --- PLOT 3: RUL Degradation Curve (The "Physics" Plot) ---
    # We sort by actual RUL to show the degradation trend clearly
    print("generating plot 3: degradation alignment...")
    sorted_indices = np.argsort(actuals)[::-1] # sort descending
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[sorted_indices], label='Ideal Degradation', color='black', linewidth=2)
    plt.scatter(range(len(preds)), preds[sorted_indices], label='Model Predictions', color='red', s=1, alpha=0.5)
    plt.title('Model Sensitivity to Degradation Regimes', fontsize=14)
    plt.xlabel('Samples (Sorted by RUL)')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('03_degradation_regime.png')
    plt.close()

    print("success! generated 3 charts.")

if __name__ == "__main__":
    evaluate()
