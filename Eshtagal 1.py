# =========================================================
# Wine dataset — Phase 1 + Phase 2 (Follow Section Steps)
# 1) EDA + Report
# 2) Cleaning (fix column names / handle missing if exists)
# 3) Pearson Correlation (feature vs target)
# 4) Chi-square (after binning numeric -> categorical)
# 5) Spearman rank correlation (feature vs target)
# 6) Gini Index (Gini gain ranking)
# =========================================================

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2
from ydata_profiling import ProfileReport


# Step 0) Load + Clean Column Names
# =========================
def load_data(path: str, target_col: str = "class") -> pd.DataFrame:
    
    # Read CSV file
    df = pd.read_csv("C:/Users/dell/Downloads/Wine dataset.csv")
    
    # Remove extra spaces from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )
    
    return df


# =========================
# Step 1) EDA + Report
# =========================
def eda_report(df: pd.DataFrame, target_col: str) -> None:
    print("======== EDA REPORT ========")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())

    print("\n-- Missing values (top 20) --")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\n-- Dtypes --")
    print(df.dtypes)

    print("\n-- Target distribution --")
    print(df[target_col].value_counts(dropna=False).sort_index())

    print("\n-- Numeric summary --")
    print(df.select_dtypes(include=[np.number]).describe().T)

    print("======== END EDA ========\n")


# =========================
# Step 2) Cleaning (safe, won't break)
# =========================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there are missing values, fill numeric columns with median.
    (This dataset has no missing values, but this keeps the script robust.)
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    return out


# =========================
# Step 3) Pearson correlation (feature vs target)
# NOTE: Pearson requires numeric target
# =========================
def pearson_feature_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If target is not numeric, convert to category codes
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    return X.corrwith(y).sort_values(key=lambda v: v.abs(), ascending=False)


def multicollinearity_pairs(df: pd.DataFrame, target_col: str, thresh: float = 0.8) -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    corr = X.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature1", "level_1": "feature2", 0: "abs_corr"})
    )

    return pairs[pairs["abs_corr"] > thresh].sort_values("abs_corr", ascending=False)


# =========================
# Step 4) Chi-square (numeric -> binning -> OHE -> chi2)
# =========================
def chi_square_after_binning(
    df: pd.DataFrame,
    target_col: str,
    q_bins: int = 4,
    top_n: int = 20
) -> pd.Series:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert each numeric feature into categorical bins using quantiles
    # duplicates="drop" prevents errors when there are repeated bin edges
    X_binned = pd.DataFrame(index=X.index)
    for col in X.columns:
        X_binned[col] = pd.qcut(X[col], q=q_bins, duplicates="drop").astype(str)

    # One-hot encode the bins
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xo = ohe.fit_transform(X_binned)
    feat_names = ohe.get_feature_names_out(X_binned.columns)

    # Ensure y is integer-coded (multi-class OK)
    if not pd.api.types.is_numeric_dtype(y):
        y_codes = y.astype("category").cat.codes
    else:
        y_codes = y.astype(int)

    scores, _ = chi2(Xo, y_codes)
    chi_scores = pd.Series(scores, index=feat_names).sort_values(ascending=False)

    return chi_scores.head(top_n)


# =========================
# Step 5) Spearman rank correlation (feature vs target)
# =========================
def spearman_feature_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    scores = {}
    for col in X.columns:
        rho, _ = spearmanr(X[col].to_numpy(), y.to_numpy(), nan_policy="omit")
        scores[col] = rho

    return pd.Series(scores).sort_values(key=lambda v: v.abs(), ascending=False)


# =========================
# Step 6) Gini Index (Gini gain ranking, median split)
# =========================
def gini_impurity(arr: np.ndarray) -> float:
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def gini_gain_for_feature(df: pd.DataFrame, feature: str, target_col: str) -> float:
    y_parent = df[target_col].to_numpy()
    parent = gini_impurity(y_parent)

    thr = np.nanmedian(df[feature].to_numpy())
    left = df[df[feature] <= thr]
    right = df[df[feature] > thr]

    n = len(df)
    weighted = 0.0
    if len(left) > 0:
        weighted += (len(left) / n) * gini_impurity(left[target_col].to_numpy())
    if len(right) > 0:
        weighted += (len(right) / n) * gini_impurity(right[target_col].to_numpy())

    return parent - weighted


def gini_rank_features(df: pd.DataFrame, target_col: str) -> pd.Series:
    X = df.drop(columns=[target_col])
    gains = {col: gini_gain_for_feature(df, col, target_col) for col in X.columns}
    return pd.Series(gains).sort_values(ascending=False)


# =========================
# RUN (edit only the paths if needed)
# =========================
if __name__ == "__main__":
    csv_path = "/mnt/data/Wine dataset.csv"
    target_col = "class"

    df = load_data(csv_path, target_col)
    df = clean_df(df)

    # 1) EDA
    eda_report(df, target_col)

    # 3) Pearson
    pearson_scores = pearson_feature_target(df, target_col)
    print("======== PEARSON (Top 10) ========")
    print(pearson_scores.head(10), "\n")

    # Pearson multicollinearity (|corr| > 0.8)
    pairs = multicollinearity_pairs(df, target_col, thresh=0.8)
    print("======== MULTICOLLINEARITY PAIRS (|corr|>0.8) ========")
    print(pairs if not pairs.empty else "No pairs above threshold.", "\n")

    # 4) Chi-square (after binning)
    chi_top = chi_square_after_binning(df, target_col, q_bins=4, top_n=15)
    print("======== CHI-SQUARE after binning (Top 15) ========")
    print(chi_top, "\n")

    # 5) Spearman
    spearman_scores = spearman_feature_target(df, target_col)
    print("======== SPEARMAN (Top 10) ========")
    print(spearman_scores.head(10), "\n")

    # 6) Gini gain ranking
    gini_scores = gini_rank_features(df, target_col)
    print("======== GINI GAIN (Top 10) ========")
    print(gini_scores.head(10), "\n")