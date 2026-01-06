#!/usr/bin/env python3

"""ipl_pipeline.py

A compact pipeline for:
- Loading matches.csv
- Handling missing 'city' values (using venue -> city mapping where possible)
- Label-encoding team columns
- Calculating average first-innings score by venue and merging it into the main df
- Creating a 'toss_decision_impact' feature (venue-level difference in win rates when batting vs bowling first)
- Splitting into train/test (80/20)
- Training LogisticRegression and XGBoost classifiers
- Evaluating with confusion matrix and accuracy

Adjust columns/paths as needed for your exact dataset schema.
"""

import os
import sys
import warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
    warnings.warn("xgboost is not installed. Install it to enable XGBoost model training.")

RANDOM_STATE = 42


def load_matches(path: str = "matches.csv") -> pd.DataFrame:
    """Load matches CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def fill_missing_cities(df: pd.DataFrame, city_col: str = "city", venue_col: str = "venue") -> pd.DataFrame:
    """
    Fill missing 'city' values by mapping from venue -> city using non-missing entries.
    Remaining missing cities are set to 'Unknown'.
    """
    if city_col not in df.columns or venue_col not in df.columns:
        # Nothing to do if columns not present
        return df

    # Build a venue -> city map from rows where city is present
    mapping = (df.loc[df[city_col].notna(), [venue_col, city_col]]
               .drop_duplicates(subset=[venue_col])
               .set_index(venue_col)[city_col]
               .to_dict())

    # Fill missing cities using mapping
    df[city_col] = df.apply(
        lambda row: mapping.get(row[venue_col], row[city_col]) if pd.isna(row[city_col]) else row[city_col],
        axis=1
    )

    # Final fallback
    df[city_col] = df[city_col].fillna("Unknown")
    return df


def compute_avg_first_innings_by_venue(df: pd.DataFrame,
                                       venue_col: str = "venue") -> pd.DataFrame:
    """
    Calculate average first-innings score at each venue and return a DataFrame with columns:
      - venue
      - avg_first_innings_score

    The function tries to infer the first-innings score from common column patterns:
      1) If the dataframe contains 'inning' and 'runs' (per-innings rows), it will use rows where inning == 1 and mean of runs.
      2) If the dataframe has a column like 'first_innings_score' or 'inning1_score', it will use that directly.
      3) If none of those exist, returns an empty dataframe.
    """
    # Case 1: per-innings rows
    if "inning" in df.columns and "runs" in df.columns and venue_col in df.columns:
        first_innings = df[df["inning"] == 1].groupby(venue_col)["runs"].mean().reset_index()
        first_innings.columns = [venue_col, "avg_first_innings_score"]
        return first_innings

    # Case 2: aggregated first-inings score column
    for col in ("first_innings_score", "inning1_score", "first_innings_runs"):
        if col in df.columns and venue_col in df.columns:
            agg = df.groupby(venue_col)[col].mean().reset_index()
            agg.columns = [venue_col, "avg_first_innings_score"]
            return agg

    # If dataset does not include per-innings scores, return empty df
    return pd.DataFrame(columns=[venue_col, "avg_first_innings_score"])


def merge_avg_score(df: pd.DataFrame,
                    avg_score_df: pd.DataFrame,
                    venue_col: str = "venue") -> pd.DataFrame:
    """Merge the averaged first innings score into the main df on venue."""
    if avg_score_df.empty:
        # No merging possible
        df["avg_first_innings_score"] = np.nan
        return df
    merged = df.merge(avg_score_df, on=venue_col, how="left")
    merged["avg_first_innings_score"] = merged["avg_first_innings_score"].fillna(merged["avg_first_innings_score"].mean())
    return merged


def create_toss_decision_impact(df: pd.DataFrame,
                                venue_col: str = "venue",
                                toss_decision_col: str = "toss_decision",
                                toss_winner_col: str = "toss_winner",
                                winner_col: str = "winner") -> pd.DataFrame:
    """
    For each venue, compute:
      bat_win_rate = proportion of matches where the team that chose 'bat' at toss went on to win
      field_win_rate = proportion of matches where the team that chose 'field' at toss went on to win

    Create a feature 'toss_decision_impact' = bat_win_rate - field_win_rate and attach it to each row.
    """
    required = {venue_col, toss_decision_col, toss_winner_col, winner_col}
    if not required.issubset(df.columns):
        # Not enough info to compute; fill with 0
        df["toss_decision_impact"] = 0.0
        return df

    # Consider only rows where toss_decision is 'bat' or 'field' (or 'bowl' variants)
    df_local = df.copy()
    df_local[toss_decision_col] = df_local[toss_decision_col].str.lower()

    # Normalize common synonyms
    df_local[toss_decision_col] = df_local[toss_decision_col].replace({"bowl": "field"})

    # Indicator whether toss_winner also won the match
    df_local["_toss_winner_won"] = (df_local[toss_winner_col] == df_local[winner_col]).astype(int)

    # Compute rates per venue & toss_decision
    pivot = (df_local.groupby([venue_col, toss_decision_col])["_toss_winner_won"]
             .agg(["mean", "count"])
             .reset_index()
             .rename(columns={"mean": "win_rate", "count": "n_matches"}))

    # Extract bat and field win rates per venue
    bat_rates = pivot[pivot[toss_decision_col] == "bat"][[venue_col, "win_rate"]].set_index(venue_col)["win_rate"]
    field_rates = pivot[pivot[toss_decision_col] == "field"][[venue_col, "win_rate"]].set_index(venue_col)["win_rate"]

    # Combine into a DataFrame
    venues = sorted(set(df_local[venue_col].unique()))
    impact_list = []
    for v in venues:
        bat = float(bat_rates.get(v, np.nan))
        fld = float(field_rates.get(v, np.nan))
        # If one is NaN, replace with 0.0 to avoid NaN impact (you may prefer other strategies)
        if np.isnan(bat):
            bat = 0.0
        if np.isnan(fld):
            fld = 0.0
        impact_list.append((v, bat - fld))

    impact_df = pd.DataFrame(impact_list, columns=[venue_col, "toss_decision_impact"])
    merged = df.merge(impact_df, on=venue_col, how="left")
    merged["toss_decision_impact"] = merged["toss_decision_impact"].fillna(0.0)
    return merged


def encode_teams(df: pd.DataFrame, team_cols: Tuple[str, str] = ("team1", "team2")) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode team name columns and return the encoders for reuse."""
    encoders = {}
    df_encoded = df.copy()
    for col in team_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
        else:
            raise KeyError(f"Team column '{col}' not found in DataFrame.")
    return df_encoded, encoders


def prepare_target(df: pd.DataFrame, team1_col: str = "team1", winner_col: str = "winner") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create a binary target: 1 if team1 won, 0 otherwise.
    The function handles cases where 'winner' may have team names (string) or encoded labels.
    """
    if winner_col not in df.columns or team1_col not in df.columns:
        raise KeyError("Required columns for target creation not found.")

    # If winner column is string team names while team1 is encoded integers, we should align them.
    y = None
    if df[winner_col].dtype == df[team1_col].dtype:
        y = (df[winner_col] == df[team1_col]).astype(int)
    else:
        # Attempt to map string winners to team1 labels if possible
        # If winner is string and team1 integers we assume encoders were applied to team columns earlier and 'winner' is still text.
        # Try to infer mapping from original team names if present. Otherwise, create a fallback by comparing names.
        if df[winner_col].dtype == object:
            # If winner is text but team1 numeric, try to find team name columns to map
            # We assume the original dataset had team1/team2 names — user should encode before calling this or ensure winner matches encoding.
            # For safety, create y by comparing winner to team1 as strings where possible:
            y = (df[winner_col].astype(str) == df[team1_col].astype(str)).astype(int)
        else:
            # Fallback numeric compare
            y = (df[winner_col] == df[team1_col]).astype(int)

    return y, df


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Split features and target into train/test with 80/20 ratio (default)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)


def train_logistic_regression(X_train, y_train, X_val=None, y_val=None) -> LogisticRegression:
    """Initialize and train a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None,
                  params: dict = None) -> "XGBClassifier":
    """Train an XGBoost classifier with given hyperparameters."""
    if XGBClassifier is None:
        raise RuntimeError("XGBoost not available. Install xgboost to use this function.")

    default_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "model"):
    """Print confusion matrix and accuracy for the test set predictions."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== Evaluation: {model_name} ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print()


def build_feature_matrix(df: pd.DataFrame,
                         feature_cols: list = None,
                         drop_cols: list = None) -> pd.DataFrame:
    """
    Construct a feature matrix from provided feature columns. If feature_cols is None,
    use a sensible default subset of numeric columns:
      - team1, team2 (assumed encoded)
      - toss_decision_impact
      - avg_first_innings_score
    """
    df_local = df.copy()
    if feature_cols is None:
        feature_cols = []
        for col in ("team1", "team2", "toss_decision_impact", "avg_first_innings_score"):
            if col in df_local.columns:
                feature_cols.append(col)

    X = df_local[feature_cols].copy()
    # Fill any remaining NaNs
    X = X.fillna(X.mean().to_dict())
    return X


def main(csv_path: str = "matches.csv"):
    # 1) Load
    df = load_matches(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 2) Fill missing cities
    df = fill_missing_cities(df, city_col="city", venue_col="venue")
    print("Filled missing city values (venue-based mapping where possible).")

    # 3) Compute avg first innings by venue and merge
    avg_score_df = compute_avg_first_innings_by_venue(df, venue_col="venue")
    if not avg_score_df.empty:
        print("Computed average first-innings scores by venue.")
    else:
        print("No explicit first-innings score info found; avg_first_innings_score will be NaN/filled with overall mean.")
    df = merge_avg_score(df, avg_score_df, venue_col="venue")

    # 4) Create toss_decision_impact feature
    df = create_toss_decision_impact(df, venue_col="venue",
                                     toss_decision_col="toss_decision",
                                     toss_winner_col="toss_winner",
                                     winner_col="winner")
    print("Created 'toss_decision_impact' feature.")

    # 5) Encode teams
    try:
        df_encoded, encoders = encode_teams(df, team_cols=("team1", "team2"))
        print("Encoded team1 and team2 into numeric labels.")
    except KeyError as e:
        print(f"Encoding error: {e}")
        return

    # 6) Prepare target: 1 if team1 won else 0
    # Note: If 'winner' is still a string, this function attempts basic alignment.
    y, df_with_target = prepare_target(df_encoded, team1_col="team1", winner_col="winner")
    print("Prepared binary target (team1 win = 1).")

    # 7) Build feature matrix
    X = build_feature_matrix(df_with_target)
    print(f"Feature matrix shape: {X.shape}")

    # 8) Train/test split 80/20
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Split data: train={len(X_train)}, test={len(X_test)}")

    # 9) Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")

    # 10) Train XGBoost if available
    if XGBClassifier is not None:
        # we use a small validation split from train for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        }
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, params=xgb_params)
        evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    else:
        print("Skipping XGBoost training because xgboost is not installed.")

    print("Pipeline complete.")


if __name__ == "__main__":
    csv = "matches.csv"
    if len(sys.argv) > 1:
        csv = sys.argv[1]
    if not os.path.exists(csv):
        print(f"File not found: {csv}. Please provide the correct path to matches.csv")
    else:
        main(csv)
