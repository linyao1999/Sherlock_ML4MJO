import torch.nn as nn
import torch
from cnn import CNN_one
from mlp import MLP

class UNet_A(nn.Module):
    def __init__(self, cnn_config, mlp_config):
        super(UNet_A, self).__init__()

        # --- Extraction of HPO Parameters ---
        num_filters_enc = cnn_config.get("num_filters_enc", 64)
        num_filters_dec1 = int(num_filters_enc * 2)
        num_filters_dec2 = int(num_filters_enc * 3)
        kernel_size_list = cnn_config.get("kernel_size", [5, 5])
        
        # Norm configuration for HPO
        # These will be suggested by Optuna: 'batch' or 'group'
        norm_type = cnn_config.get("norm_type", "batch") 
        num_groups = cnn_config.get("num_groups", 8)

        # Helper to pass consistent norm params to CNN_one
        cnn_kwargs = {
            "kernel_size": (kernel_size_list[0], kernel_size_list[1]),
            "stride": cnn_config.get("stride", 1),
            "padding": cnn_config.get("padding", 'same'),
            "dropout": cnn_config.get("dropout", 0.0), # Set to 0.0 per your rules
            "norm_type": norm_type,
            "num_groups": num_groups
        }

        # --- Architecture Definition ---
        self.hid1 = CNN_one(input_channel_num=cnn_config["input_channel_num"], 
                            output_channel_num=num_filters_enc, **cnn_kwargs)
        
        self.hid2 = CNN_one(input_channel_num=num_filters_enc, 
                            output_channel_num=num_filters_enc, **cnn_kwargs)
        
        self.hid3 = CNN_one(input_channel_num=num_filters_enc, 
                            output_channel_num=num_filters_enc, **cnn_kwargs)
        
        self.hid4 = CNN_one(input_channel_num=num_filters_enc, 
                            output_channel_num=num_filters_enc, **cnn_kwargs)
        
        self.hid5 = CNN_one(input_channel_num=num_filters_enc, 
                            output_channel_num=num_filters_enc, **cnn_kwargs)

        self.hid6 = CNN_one(input_channel_num=num_filters_dec1, 
                            output_channel_num=num_filters_dec1, **cnn_kwargs)

        # Dynamically calculate flatten size for MLP
        flatten_size = num_filters_dec2 * cnn_config['input_map_size']

        self.mlp = MLP(
            input_size=flatten_size,
            hidden_layers=mlp_config["hidden_layers"],
            output_size=mlp_config["output_size"],
            dropout=mlp_config.get("dropout", 0.5),
            use_layer_norm=True # Recommended for MJO index regression
        )

    def forward(self, x):
        x1 = self.hid1(x)
        x2 = self.hid2(x1)
        x3 = self.hid3(x2)
        x4 = self.hid4(x3)
        x5 = self.hid5(x4)

        x3p5 = torch.cat((x5, x3), dim=1) # Concatenation 1
        x6 = self.hid6(x3p5)

        x2p6 = torch.cat((x6, x2), dim=1) # Concatenation 2
        x = x2p6.view(x2p6.size(0), -1)
        
        return self.mlp(x)
        