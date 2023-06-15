import pandas as pd
from ydata_profiling import ProfileReport


def eda(parquet_path, report_name):
    df = pd.read_parquet(parquet_path)
    profile = ProfileReport(df, title=report_name)
    profile.to_file(f"{report_name}.html")


eda("./data/raw_data/phase-1/prob-1/raw_train.parquet", "prob-1")
eda("./data/raw_data/phase-1/prob-2/raw_train.parquet", "prob-2")
