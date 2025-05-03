import torch.nn as nn

# Deeper tuned MLP model with BatchNorm and no final Sigmoid
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Batch normalization 
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 1)     # No Sigmoid here; we'll apply it in the loss function
        )

    def forward(self, x):
        return self.net(x)