
import os
import yaml
import torch
from experiment import run_experiment

CONFIG_PATH = "configs/transformer_memory.yaml"  # <- point to large_mlp config

def main():
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = run_experiment(config, device)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/continual_rl_results.json", index=False)

    print("\n✅ memory results saved to results/transformer_memory_results.csv")

if __name__ == "__main__":
    main()
