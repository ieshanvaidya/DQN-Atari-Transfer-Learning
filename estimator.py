import torch.nn as nn


class Estimator(nn.Module):
    def __init__(self, num_actions, agent_history_length=4):
        """
        Estimator class; returns Q-values
        """

        super(Estimator, self).__init__()

        self.model = nn.Sequential(
            # Input: batch x m x 84 x 84
            nn.Conv2d(in_channels=agent_history_length, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Input: batch x 32 x 20 x 20
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Input: batch x 64 x 9 x 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Input: batch x 64 x 7 x 7
            nn.Flatten(1),
            # Input: batch x 3136
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            # Input: batch x 512
            nn.Linear(in_features=512, out_features=num_actions)
            )

    def forward(self, x):
        out = self.model(x)
        return out
