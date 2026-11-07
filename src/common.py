"""Shared data loading, splitting and model definitions for the banknote project.

Dataset: UCI Banknote Authentication (Lohweg, 2012), CC BY 4.0.
https://archive.ics.uci.edu/dataset/267/banknote+authentication
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "data_banknote_authentication.txt"
SPLIT_FILE = ROOT / "data" / "split.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 5420
FEATURES = ["variance", "skewness", "kurtosis", "entropy"]
TARGET = "forged"  # class 1 in the UCI file; see Issues in baseline.txt about label coding
TEST_FRAC = 0.20
N_FOLDS = 5

# Hyperparameter grids (log grid for C, as in the proposal feedback)
C_GRID = np.logspace(-3, 3, 13)
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31]
RF_MAX_FEATURES = [1, 2, 4]
RF_MIN_LEAF = [1, 3]
RF_TREES = 200


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, header=None, names=FEATURES + [TARGET])
    df.insert(0, "row_id", np.arange(len(df)))  # original file order, kept for leakage checks
    return df


def load_clean():
    """Return (clean dataframe, number of exact duplicate rows dropped)."""
    raw = load_raw()
    dup = raw.duplicated(subset=FEATURES + [TARGET], keep="first")
    return raw.loc[~dup].reset_index(drop=True), int(dup.sum())


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 80/20 dev/test split, saved to disk so every script uses the same one."""
    dev_idx, test_idx = train_test_split(
        df.index, test_size=TEST_FRAC, stratify=df[TARGET], random_state=SEED
    )
    split = pd.DataFrame({"row_id": df["row_id"], "split": "dev"})
    split.loc[test_idx, "split"] = "test"
    split.to_csv(SPLIT_FILE, index=False)
    return split


def load_dev():
    """Development data only. The test rows are never returned by this function."""
    df, _ = load_clean()
    split = pd.read_csv(SPLIT_FILE) if SPLIT_FILE.exists() else make_split(df)
    dev_ids = set(split.loc[split["split"] == "dev", "row_id"])
    dev = df[df["row_id"].isin(dev_ids)].reset_index(drop=True)
    return dev


def pseudo_specimen_groups(df: pd.DataFrame, block: int = 8) -> np.ndarray:
    """Heuristic group ids: consecutive runs of `block` rows (in original file order)
    within each class. Used only for the leakage sensitivity check, because the UCI
    file has no specimen ids."""
    groups = np.empty(len(df), dtype=int)
    next_id = 0
    for c in (0, 1):
        idx = df.index[df[TARGET] == c]
        order = idx[np.argsort(df.loc[idx, "row_id"].values)]
        g = np.arange(len(order)) // block + next_id
        groups[order] = g
        next_id = g.max() + 1
    return groups


def models():
    """(name, pipeline, param_grid) for each method. Scaling sits inside the
    pipeline so it is re-fitted on each training fold."""
    return {
        "Logistic regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": C_GRID},
        ),
        "kNN": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": K_GRID},
        ),
        "Random forest": (
            RandomForestClassifier(n_estimators=RF_TREES, random_state=SEED, n_jobs=1),
            {"max_features": RF_MAX_FEATURES, "min_samples_leaf": RF_MIN_LEAF},
        ),
    }
