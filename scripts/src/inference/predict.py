import torch
import torch.nn as nn
from collections import defaultdict

def predict(model, data_loader, device="cuda"):
    """
    Standard prediction loop for RMM/ROMI indices[cite: 1288, 1289].
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch 
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
            
    return torch.cat(predictions, dim=0)

def register_norm_hooks(model, feature_maps_dict):
    """
    Registers hooks to capture feature maps AFTER the normalization step.
    If no normalization is present for a layer, it defaults to the Conv2d output.
    """
    handles = []

    def hook_fn(name):
        def fn(module, input, output):
            # Save as CPU tensor to ensure GPU memory is preserved for inference [cite: 1480]
            feature_maps_dict[name] = output.detach().cpu()
        return fn

    # We iterate through named modules to find the layers after processing
    for name, module in model.named_modules():
        # Target the normalization layers specifically (BatchNorm or GroupNorm)
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            handles.append(module.register_forward_hook(hook_fn(name)))
        
        # Fallback: if a layer is a Conv2d and it's NOT followed by a norm in your 
        # specific architecture (like a final layer), you might still want it.
        # But for UNet_A, capturing the Norm layer is the "After Norm" signal.

    return handles

def predict_with_features(model, data_loader, device="cuda"):
    """
    Inference with feature map collection for spectral analysis[cite: 342, 519].
    Captures the representation after the normalization stabilization.
    """
    model.eval()
    predictions = []
    feature_map_batches = defaultdict(list)
    feature_maps_storage = {}

    # Register hooks ONCE for all normalization layers
    handles = register_norm_hooks(model, feature_maps_storage)

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            predictions.append(outputs.cpu())
            
            # Transfer batch feature maps to accumulator [cite: 1481]
            for lname, fmap in feature_maps_storage.items():
                feature_map_batches[lname].append(fmap)
    
    # Remove hooks
    for h in handles:
        h.remove()

    # Stack along time dimension for spectral projection analysis [cite: 1389]
    final_feature_maps = {k: torch.cat(v, dim=0) for k, v in feature_map_batches.items()}
    final_predictions = torch.cat(predictions, dim=0)
    
    return final_predictions, final_feature_maps
