import torch.nn as nn
import torch.nn.functional as F

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, **kwargs):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
        
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='constant', value=0)
        
        return self.conv(x)

class CNN_one(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, kernel_size=5, 
                 stride=1, padding='same', dropout=0.2, norm_type='batch', num_groups=8):
        super(CNN_one, self).__init__()

        if padding == 'same':
            if isinstance(kernel_size, (list, tuple)):
                actual_padding = (kernel_size[1] - 1) // 2
            else:
                actual_padding = (kernel_size - 1) // 2
        else:
            actual_padding = padding

        layers = []
        layers.append(
            PeriodicConv2d(
                in_channels=input_channel_num,
                out_channels=output_channel_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=actual_padding 
            )
        )

        # Updated Normalization Logic for Optuna HPO
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(output_channel_num))
        elif norm_type == 'group':
            # Note: output_channel_num must be divisible by num_groups
            layers.append(nn.GroupNorm(num_groups, output_channel_num))
        # if norm_type is None or 'none', no layer is added
        
        layers.append(nn.ReLU())

        # Note: Your study utilizes Monte Carlo Dropout (MCDO) to mitigate overfitting [cite: 759]
        layers.append(nn.Dropout2d(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_channel_num, channels_list, kernel_size=5, 
                 stride=1, padding='same', dropout=0.2, norm_type='batch', num_groups=8):
        """
        Supports HPO tuning for normalization type and GroupNorm groups.
        """
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channel_num

        for out_channels in channels_list:
            layers.append(
                CNN_one(
                    input_channel_num=in_channels,
                    output_channel_num=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    norm_type=norm_type,
                    num_groups=num_groups
                )
            )
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

        