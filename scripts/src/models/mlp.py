import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, use_layer_norm=True):
        """
        Updated MLP for MJO index regression.
        Includes Layer Normalization to stabilize RMM/ROMI amplitude prediction.
        """
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        
        for hidden in hidden_layers:
            # Linear Layer
            layers.append(nn.Linear(in_features, hidden))
            
            # Layer Normalization (Added for stability in 2026 SOTA models)
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            
            # Activation and Dropout
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_features = hidden
        
        # Output layer for RMM1/RMM2 or ROMI1/ROMI2
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flattening is handled here to ensure compatibility with CNN output
        x = x.view(x.size(0), -1) 
        return self.network(x)

