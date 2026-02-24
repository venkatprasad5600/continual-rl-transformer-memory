# Problem Statement

Standard PPO agents forget previously learned tasks when trained sequentially.

This project explores:

Can temporal modeling and episodic memory reduce forgetting in continual RL?

# 🧠 Models Implemented
Model	Description
PPO-MLP	Baseline 2-layer MLP
PPO-Large-MLP	Higher capacity MLP
PPO-Transformer	Sequence modeling without explicit memory
PPO-Memory	PPO + Episodic Replay Buffer
PPO-Transformer-Memory	Transformer + Episodic Memory

# Experimental Setup

Environment: MiniGrid Sequential Tasks
Optimizer: Adam
Algorithm: PPO

Rollout-based training
Single-seed controlled experiment (seed=0)
Sequential training across 3 tasks

# 🔍 Key Findings

Increasing MLP capacity improves retention slightly.
Transformer improves long-horizon representation.
Episodic memory significantly reduces forgetting.
Transformer + Memory achieves the best forward transfer and retention.
The hybrid architecture shows the lowest catastrophic forgetting across tasks.

# 🏗 Architecture Highlights

PPO clipped objective
Transformer encoder (sequence modeling)
Fixed-length observation sequence window
Episodic memory replay buffer
Replay-based policy update
Config-driven experiment pipeline

# 🔬 Research Relevance

This project demonstrates:
Continual learning experimentation
Memory-augmented policy optimization
Sequence modeling in RL
Comparative ablation analysis
Clean experimental reproducibility


# 🛠 Tech Stack

Python
PyTorch
Gymnasium

MiniGrid

YAML Config-based experiment control
