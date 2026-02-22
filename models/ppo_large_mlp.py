
import torch
import torch.nn as nn


class PPOLargeAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):  # bigger network
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.net(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.squeeze(0), dist.log_prob(action).squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states, actions):
        """
        Compute log_probs, entropy, and value for a batch of states & actions
        """
        logits, value = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, value