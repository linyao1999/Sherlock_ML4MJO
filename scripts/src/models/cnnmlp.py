import torch.nn as nn
import torch
from .cnn import CNN
from .mlp import MLP

class CNNMLP(nn.Module):
    def __init__(self, cnn_config, mlp_config):
        """
        Designed for Simple CNN experiments (e.g., 2-layer CNN).
        Supports HPO for Batch/Group normalization and LayerNorm in the MLP.
        """
        super(CNNMLP, self).__init__()

        # --- Extraction of HPO Parameters ---
        kernel_size_list = cnn_config.get("kernel_size", [5, 5])
        
        # New normalization parameters suggested by Optuna
        norm_type = cnn_config.get("norm_type", "batch") 
        num_groups = cnn_config.get("num_groups", 8)

        # Initialize the CNN (e.g., 2 layers for your simplified experiments)
        self.cnn = CNN(
            input_channel_num=cnn_config["input_channel_num"],
            channels_list=cnn_config["channels_list"],
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.0), # Fixed to 0.0 for CNN per rules
            norm_type=norm_type,
            num_groups=num_groups
        )

        # Compute the flattened size based on the final filter count and map dimensions
        # Channels_list[-1] is used as there is no up/downsampling [cite: 1332]
        flattened_size = cnn_config['channels_list'][-1] * cnn_config['input_map_size']

        # Initialize the MLP with Layer Normalization for stable RMM/ROMI regression
        self.mlp = MLP(
            input_size=flattened_size,
            hidden_layers=mlp_config["hidden_layers"],
            output_size=mlp_config["output_size"],
            dropout=mlp_config.get("dropout", 0.5),
            use_layer_norm=True 
        )

        print(f'Initialized CNNMLP with {norm_type} norm')

    def forward(self, x):
        x = self.cnn(x)             # Pass through shallow CNN
        x = x.view(x.size(0), -1)   # Flatten (Batch, Channels * H * W)
        x = self.mlp(x)             # Final regression to MJO indices
        return x
