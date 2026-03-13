import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
from train import train_model, train_model_hpo

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.net(x)

def verify():
    print("--- Starting Verification ---")
    
    # 1. Generate Synthetic Data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1) 
    dataset = TensorDataset(X, y)
    
    # Correct way to split datasets in PyTorch
    train_ds, val_ds = random_split(dataset, [800, 200])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device detected: {device}")

    # 2. Test Standard Training
    print("\nTesting train_model...")
    config = {
        "training": {
            "epochs": 3,
            "learning_rate": 0.1, # High LR to trigger scheduler quickly
            "optimizer": "AdamW",
            "early_stopping_patience": 5
        }
    }
    
    try:
        model = DummyModel()
        trained_model = train_model(model, train_loader, val_loader, config)
        print("✅ train_model completed successfully.")
    except Exception as e:
        print(f"❌ train_model failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test HPO Training (Optuna)
    print("\nTesting train_model_hpo...")
    def objective(trial):
        hpo_model = DummyModel()
        # Suggest a learning rate to simulate a real trial
        config["training"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        loss, _ = train_model_hpo(hpo_model, train_loader, val_loader, config, trial)
        return loss

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(objective, n_trials=1)
        print("✅ train_model_hpo completed successfully.")
    except Exception as e:
        print(f"❌ train_model_hpo failed: {e}")
        return

    print("\n--- All checks passed! ---")

if __name__ == "__main__":
    verify()
    