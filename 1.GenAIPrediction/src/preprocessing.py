import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, SELECTED_FEATURES, TEST_SIZE


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_bi = df.copy()
    df_bi["diagnostico_binario"] = df_bi["diagnostico"].map({"M": 1, "B": 0})
    X = df_bi[SELECTED_FEATURES]
    y = df_bi["diagnostico_binario"]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
