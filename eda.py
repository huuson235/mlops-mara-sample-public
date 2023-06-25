import pandas as pd
from ydata_profiling import ProfileReport


def eda(x_parquet_path, y_parquet_path, report_name):
    df = pd.read_parquet(x_parquet_path)
    y = pd.read_parquet(y_parquet_path)
    df["label"] = y["label"]
    profile = ProfileReport(df, title=report_name)
    profile.to_file(f"{report_name}.html")


eda("./data/train_data/phase-1/prob-1/train_x.parquet", "./data/train_data/phase-1/prob-1/train_y.parquet", "train-prob-1")
eda("./data/train_data/phase-1/prob-2/train_x.parquet", "./data/train_data/phase-1/prob-2/train_y.parquet", "train-prob-2")

# eda("./data/raw_data/phase-1/prob-2/raw_train.parquet", "prob-2")
