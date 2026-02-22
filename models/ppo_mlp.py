
# models/ppo_mlp.py

import torch
import torch.nn as nn

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.net(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, state):
        # convert to batch if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.squeeze(0), dist.log_prob(action).squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states, actions):
        logits, value = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, value