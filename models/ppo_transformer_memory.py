import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOTransformerMemoryAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, config=None):
        super().__init__()

        # ===== Config =====
        if config and "transformer" in config:
            tcfg = config["transformer"]
            d_model = tcfg.get("d_model", 128)
            n_heads = tcfg.get("n_heads", 4)
            n_layers = tcfg.get("n_layers", 2)
            self.seq_len = tcfg.get("seq_len", 8)
        else:
            d_model = 128
            n_heads = 4
            n_layers = 2
            self.seq_len = 8

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ===== Layers =====
        self.input_proj = nn.Linear(obs_dim, d_model)

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

        # ===== Sequence buffer =====
        self.sequence = []

    # =====================================================
    # SEQUENCE HANDLING
    # =====================================================

    def reset_sequence(self):
        self.sequence = []

    def update_sequence(self, obs):
        # obs must be (obs_dim,)
        if obs.dim() > 1:
            obs = obs.squeeze(0)

        self.sequence.append(obs.detach())

        if len(self.sequence) > self.seq_len:
            self.sequence.pop(0)

    def get_sequence_tensor(self, device):
        """
        Returns tensor of shape:
        (seq_len, obs_dim)
        """

        if len(self.sequence) == 0:
            return torch.zeros(self.seq_len, self.obs_dim, device=device)

        seq = torch.stack(self.sequence)  # (current_len, obs_dim)

        if seq.size(0) < self.seq_len:
            pad_len = self.seq_len - seq.size(0)
            pad = torch.zeros(pad_len, self.obs_dim, device=device)
            seq = torch.cat([pad, seq], dim=0)

        return seq  # (seq_len, obs_dim)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, seq):
        """
        seq expected shape:
        (seq_len, obs_dim)
        """

        if seq.dim() == 3:
            # already batched
            x = seq
        else:
            x = seq.unsqueeze(0)  # (1, seq_len, obs_dim)

        x = self.input_proj(x)
        x = self.transformer(x)

        x_last = x[:, -1, :]  # last timestep

        logits = self.policy_head(x_last)
        value = self.value_head(x_last).squeeze(-1)

        return logits, value

    # =====================================================
    # ACTION
    # =====================================================

    def get_action(self, obs):
        device = obs.device

        # obs must be (obs_dim,)
        if obs.dim() > 1:
            obs = obs.squeeze(0)

        self.update_sequence(obs)

        seq = self.get_sequence_tensor(device)  # (seq_len, obs_dim)

        logits, value = self.forward(seq)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    # =====================================================
    # PPO UPDATE
    # =====================================================

    def evaluate_actions(self, states, actions):
        """
        states shape:
        (batch, obs_dim)

        We fake sequences during PPO update.
        """

        batch_size = states.size(0)

        # Expand states into fake sequences
        seq = states.unsqueeze(1).repeat(1, self.seq_len, 1)

        logits, values = self.forward(seq)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, entropy, values
