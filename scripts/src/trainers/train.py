import torch
import torch.nn as nn
import torch.optim as optim
import math
import optuna
import numpy as np

def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_optimizer_and_scheduler(model, training_config):
    """
    Sets up the optimizer and scheduler based on config choices.
    """
    opt_name = training_config.get("optimizer", "AdamW")
    lr = training_config.get("learning_rate", 1e-3)
    weight_decay = training_config.get("weight_decay", 1e-4)
    
    if opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Standard for Adam: Reduce LR when validation plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif opt_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=training_config.get("momentum", 0.9),
            nesterov=True
        )
        # Best for SGD: Smooth decay to find flat minima
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.get("epochs", 20)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
        
    return optimizer, scheduler

def train_model(model, train_loader, val_loader, config):
    device = get_device()
    model.to(device)

    # Load pretrained weights if provided
    pretrained_path = config.get("pretrained_path")
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

    training_config = {
        "learning_rate": 1e-3,
        "epochs": 20,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "early_stopping_patience": 10,
        **config.get("training", {})
    }

    criterion = getattr(nn, training_config["criterion"])()
    optimizer, scheduler = get_optimizer_and_scheduler(model, training_config)
    
    # 2026 SOTA: Use Automatic Mixed Precision (AMP) for speed/memory efficiency
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(training_config["epochs"]):
        # --- Training Phase ---
        model.train()
        train_running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}. Skipping batch.")
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_running_loss += criterion(outputs, targets).item() * inputs.size(0)

        avg_val_loss = val_running_loss / len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{training_config['epochs']}] | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Scheduler Step (Handling different types)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= training_config["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def train_model_hpo(model, train_loader, val_loader, config, trial):
    """
    Optuna-compatible training loop with pruning, scheduler support, 
    and detailed loss reporting.
    """
    device = get_device()
    model.to(device)
    
    training_config = {
        "learning_rate": 1e-3,
        "epochs": 20,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "early_stopping_patience": 7, 
        **config.get("training", {})
    }
    
    criterion = getattr(nn, training_config["criterion"])()
    optimizer, scheduler = get_optimizer_and_scheduler(model, training_config)
    
    best_val_loss = float("inf")
    val_loss_history = []
    no_improve_epochs = 0

    for epoch in range(training_config["epochs"]):
        # --- Training Phase ---
        model.train()
        train_running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss
            train_running_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_running_loss += criterion(model(inputs), targets).item() * inputs.size(0)
        
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)

        # Detailed Printing for HPO Monitoring
        print(f"[Trial {trial.number} | Epoch {epoch+1}/{training_config['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Optuna Reporting & Pruning
        trial.report(avg_val_loss, step=epoch)
        if trial.should_prune() or math.isnan(avg_val_loss):
            print(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.TrialPruned()

        # Scheduler Step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Early stopping for the trial
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= training_config["early_stopping_patience"]:
            print(f"Trial {trial.number} early stopping at epoch {epoch+1}")
            break

    return best_val_loss, val_loss_history