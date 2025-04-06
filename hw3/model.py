import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hidden_sizes=[512, 256], with_baseline=False, init_std=1.0) -> None:
        super(VPG, self).__init__()
        
        self.with_baseline = with_baseline
        self.init_std = init_std
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], act_dim)
        
        if with_baseline:
            self.value_layer = nn.Sequential(
                nn.Linear(hidden_sizes[1], 1),
                nn.Tanh()
            )
        
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.constant_(self.log_std_layer.bias, np.log(self.init_std))

    def forward(self, x, return_baseline=False):
        for layer in self.feature_extractor:
            x = layer(x)
        
        action_mean = self.mean_layer(x)
        action_log_std = self.log_std_layer(x)
        
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        policy_output = torch.cat([action_mean, action_log_std], dim=-1)
        
        if self.with_baseline and return_baseline:
            value = self.value_layer(x)
            return policy_output, value
        else:
            return policy_output
    