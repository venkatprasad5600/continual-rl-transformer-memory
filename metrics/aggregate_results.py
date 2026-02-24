
import pandas as pd


def aggregate(csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)

    grouped = combined.groupby(["after_task", "eval_task"])["score"]

    mean_scores = grouped.mean()
    std_scores = grouped.std().fillna(0)

    return mean_scores, std_scores
