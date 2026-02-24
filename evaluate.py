
import torch
import numpy as np


def evaluate(agent, make_env_fn, num_episodes, device):

    agent.eval()

    episode_rewards = []

    for _ in range(num_episodes):

        env = make_env_fn()
        obs, _ = env.reset()

        # Reset sequence for transformer models
        if hasattr(agent, "reset_sequence"):
            agent.reset_sequence()

        done = False
        total_reward = 0.0

        while not done:

            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                action, _, _ = agent.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(
                action.item()
            )

            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_rewards.append(total_reward)
        env.close()

    agent.train()

    return float(np.mean(episode_rewards))
