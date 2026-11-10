"""Data loading, cleaning and preprocessing pipeline."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hr_data.csv"

TARGET = "Attrition"
# Columns with a single constant value or just an ID -> no predictive value
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the HR dataset (generates a synthetic one if the CSV is missing)."""
    path = Path(path)
    if not path.exists():
        from src.generate_data import generate

        path.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(path, index=False)

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def split_columns(X: pd.DataFrame):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical columns."""
    numeric, categorical = split_columns(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"), categorical),
        ]
    )


def get_splits(test_size: float = 0.2, seed: int = 42):
    """Return stratified train/test splits (80/20)."""
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
