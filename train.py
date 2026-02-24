
import torch
import torch.nn.functional as F
from storage.rollout_buffer import RolloutBuffer
from memory.episodic_memory import EpisodicMemory


def train_single_task(agent, env, config, device, timesteps):
    agent.train()
    cfg = config["training"]
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["learning_rate"])

    gamma = cfg["gamma"]
    lam = cfg["gae_lambda"]
    clip_eps = cfg["clip_range"]
    ppo_epochs = cfg["update_epochs"]
    batch_size = cfg["batch_size"]
    rollout_length = cfg.get("rollout_length", 128)
    memory_capacity = cfg.get("memory_capacity", 1000)
    replay_batch_size = cfg.get("replay_batch_size", 64)

    memory = EpisodicMemory(memory_capacity, device)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    total_steps = 0

    while total_steps < timesteps:
        buffer = RolloutBuffer(rollout_length, device)

        for _ in range(rollout_length):
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buffer.add(obs, action, reward, done, log_prob, value)
            memory.add(obs, action, reward, done, log_prob, value)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            total_steps += 1

            if done:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device)
                if hasattr(agent, "reset_sequence"):
                    agent.reset_sequence()

            if total_steps >= timesteps:
                break

        with torch.no_grad():
            _, _, next_value = agent.get_action(obs)

        buffer.compute_returns_and_advantages(next_value, gamma, lam)

        for _ in range(ppo_epochs):
            for states, actions, old_log_probs, returns, advantages, _ in buffer.get_batches(batch_size):
                log_probs, entropy, values = agent.evaluate_actions(states, actions)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            #  MEMORY REPLAY
            mem_batch = memory.sample(replay_batch_size)
            if mem_batch is not None:
                mem_states, mem_actions, _, _, mem_old_log_probs, mem_values = mem_batch

                log_probs, entropy, values = agent.evaluate_actions(mem_states, mem_actions)

                replay_loss = F.mse_loss(values, mem_values)

                optimizer.zero_grad()
                replay_loss.backward()
                optimizer.step()

    return agent
