
# models/ppo_transformer.py

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOTransformerAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        seq_len=8,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.seq_len = seq_len

        self.input_projection = nn.Linear(obs_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Linear(d_model, 1)

        self.sequence = []

    # ----------------------------
    # Sequence Management
    # ----------------------------
    def reset_sequence(self):
        self.sequence = []

    def update_sequence(self, obs):
        """
        obs shape: (obs_dim,)
        """
        self.sequence.append(obs)

        if len(self.sequence) > self.seq_len:
            self.sequence.pop(0)

    def get_sequence_tensor(self, device):
        """
        Returns tensor shape:
        (1, seq_len, obs_dim)
        """

        if len(self.sequence) == 0:
            return torch.zeros(
                1, self.seq_len, self.obs_dim, device=device
            )

        seq = self.sequence

        # pad if needed
        if len(seq) < self.seq_len:
            pad = [
                torch.zeros_like(seq[0])
                for _ in range(self.seq_len - len(seq))
            ]
            seq = pad + seq

        # Each element shape: (obs_dim,)
        seq_tensor = torch.stack(seq, dim=0)  # (seq_len, obs_dim)

        seq_tensor = seq_tensor.unsqueeze(0)  # (1, seq_len, obs_dim)

        return seq_tensor

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x):
        """
        x: (batch, seq_len, obs_dim)
        """

        x = self.input_projection(x)
        x = self.transformer(x)

        x = x[:, -1, :]  # last token

        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)

        return logits, value

    # ----------------------------
    # Acting
    # ----------------------------
    def get_action(self, obs):
        """
        obs shape: (obs_dim,)
        """

        device = obs.device

        # REMOVE batch dimension if exists
        if obs.dim() > 1:
            obs = obs.squeeze(0)

        self.update_sequence(obs)

        seq = self.get_sequence_tensor(device)

        logits, value = self.forward(seq)

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    # ----------------------------
    # PPO Evaluation
    # ----------------------------
    def evaluate_actions(self, states, actions):
        """
        states: (batch, obs_dim)
        """

        batch_size = states.size(0)

        # Fake sequence for PPO update
        states = states.unsqueeze(1).repeat(1, self.seq_len, 1)
        # (batch, seq_len, obs_dim)

        logits, values = self.forward(states)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, entropy, values