import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y

