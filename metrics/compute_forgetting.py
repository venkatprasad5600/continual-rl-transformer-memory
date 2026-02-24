
def compute_forgetting(df):
    forgetting = {}

    tasks = df["eval_task"].unique()

    for task in tasks:
        first_score = df[
            (df["after_task"] == task) & (df["eval_task"] == task)
        ]["score"].mean()

        last_score = df[
            (df["after_task"] == df["after_task"].max())
            & (df["eval_task"] == task)
        ]["score"].mean()

        forgetting[task] = first_score - last_score

    return forgetting
