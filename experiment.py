
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import torch
import pandas as pd

from train import train_single_task
from evaluate import evaluate
from models.ppo_transformer_memory import PPOTransformerMemoryAgent
from memory.episodic_memory import EpisodicMemory


def make_env(env_name, seed):
    env = gym.make(env_name)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def run_experiment(config, device):
    results = []

    env_names = config["envs"]
    seeds = config["seeds"]
    timesteps = config["training"]["timesteps_per_task"]
    eval_episodes = config["training"].get("eval_episodes", 10)

    for seed in seeds:
        print(f"\n===== Running Seed {seed} =====")

        temp_env = make_env(env_names[0], seed)
        obs_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.n
        temp_env.close()

        agent = PPOTransformerMemoryAgent(obs_dim, action_dim).to(device)

        #  Initialize memory ONCE per seed
        memory = EpisodicMemory(capacity=5000, device=device)

        for task_id, task_name in enumerate(env_names):
            print(f"\nTraining on Task {task_id}: {task_name}")

            agent.reset_sequence()

            env = make_env(task_name, seed)

            agent = train_single_task(
                agent,
                env,
                config,
                device,
                timesteps
            )

            # Evaluate on all seen tasks
            for eval_id in range(task_id + 1):
                agent.reset_sequence()

                score = evaluate(
                    agent,
                    lambda: make_env(env_names[eval_id], seed),
                    eval_episodes,
                    device,
                )

                results.append(
                    {
                        "seed": seed,
                        "after_task": task_id,
                        "eval_task": eval_id,
                        "score": score,
                    }
                )

            env.close()

    return pd.DataFrame(results)
