import torch.nn as nn
import torch

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)

class GatingNetwork(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.network(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.gating_network = GatingNetwork(num_experts)
    def forward(self, x):
        gate_outputs = self.gating_network(x)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, num_classes]
        output = torch.sum(expert_outputs * gate_outputs.unsqueeze(2), dim=1)  # [batch_size, num_classes]
        return output
