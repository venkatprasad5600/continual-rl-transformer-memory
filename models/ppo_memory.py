
# models/ppo_memory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOMemoryAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.shared(x)

        logits = self.policy(features)
        value = self.value(features).squeeze(-1)

        return logits, value

    def get_action(self, obs):
        logits, value = self.forward(obs)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, entropy, values