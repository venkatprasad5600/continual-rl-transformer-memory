import random
import torch


class EpisodicMemory:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.memory = []

    def add(self, state, action, reward, done, log_prob, value):
        """
        state must be (obs_dim,)
        """

        if state.dim() > 1:
            state = state.squeeze(0)

        if action.dim() == 0:
            action = action.unsqueeze(0)

        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(
            (
                state.detach().cpu(),                    # (obs_dim,)
                action.detach().cpu(),                   # (1,)
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32),
                log_prob.detach().cpu(),
                value.detach().cpu(),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            min(batch_size, len(self.memory))
        )

        states, actions, rewards, dones, log_probs, values = zip(*batch)

        states = torch.stack(states)              # (batch, obs_dim)
        actions = torch.cat(actions).long()      # (batch,)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        return (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
            log_probs.to(self.device),
            values.to(self.device),
        )

    def __len__(self):
        return len(self.memory)
