
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_length, device):
        self.rollout_length = rollout_length
        self.device = device
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        # Ensure all tensors are at least 1D
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, obs_dim]

        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long, device=self.device)
        elif action.dim() == 0:
            action = action.unsqueeze(0)

        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor([log_prob], dtype=torch.float32, device=self.device)
        elif log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)

        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], dtype=torch.float32, device=self.device)
        elif value.dim() == 0:
            value = value.unsqueeze(0)

        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, next_value, gamma, lam):
        advantages = []
        gae = torch.zeros(1, device=self.device)
        values = self.values + [torch.tensor([next_value], dtype=torch.float32, device=self.device)]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        # Stack tensors
        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.dones = torch.cat(self.dones, dim=0)
        self.log_probs = torch.cat(self.log_probs, dim=0)
        self.values = torch.cat(self.values, dim=0)
        self.advantages = torch.cat(advantages, dim=0)
        self.returns = torch.cat(returns, dim=0)

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
                self.values[batch_idx],
            )
