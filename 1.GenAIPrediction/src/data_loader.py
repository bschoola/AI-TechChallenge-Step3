import pandas as pd

from config import COLUMN_RENAME_MAP, DATA_URL


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    return df
