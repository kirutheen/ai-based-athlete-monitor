"""
Training pipeline for a learned (ML) injury-risk model.

Operates on Level B (movement-level biomechanical features) and Level C
(personalized baseline deviations) features. Strictly excludes rule-derived
output targets ('flagged', 'risk_score', 'contributing_features', 'affected_joints')
to prevent data leakage.

Enforces athlete-level train/test splitting (GroupShuffleSplit by athlete_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS = (
    "fppa_left",
    "fppa_right",
    "knee_flexion_left",
    "knee_flexion_right",
    "trunk_lean",
)

# Expanded Level B + Level C feature set for ML modeling
LEVEL_BC_FEATURE_COLUMNS = (
    "peak_knee_valgus",
    "mean_knee_valgus",
    "knee_valgus_rom",
    "knee_valgus_velocity",
    "knee_valgus_acceleration",
    "peak_knee_flexion",
    "minimum_knee_flexion",
    "knee_flexion_rom",
    "peak_hip_flexion",
    "peak_hip_abduction",
    "peak_ankle_dorsiflexion",
    "trunk_flexion",
    "trunk_lateral_flexion",
    "pelvic_obliquity",
    "knee_flexion_LSI",
    "knee_valgus_LSI",
    "knee_valgus_at_initial_contact",
    "knee_flexion_at_initial_contact",
    "valgus_duration",
    "com_displacement",
    "pose_quality",
    "baseline_deviation",
    "baseline_zscore",
)

EXCLUDED_TARGET_COLUMNS = (
    "flagged",
    "risk_score",
    "contributing_features",
    "affected_joints",
)

LABEL_COLUMN = "injured_within_90_days"


@dataclass
class TrainTestSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_athletes: set
    test_athletes: set


def athlete_level_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainTestSplit:
    """Split event-level rows into train/test by athlete_id, never by row."""
    if "athlete_id" not in df.columns:
        raise ValueError("DataFrame must have an 'athlete_id' column")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["athlete_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    train_athletes = set(train_df["athlete_id"])
    test_athletes = set(test_df["athlete_id"])
    overlap = train_athletes & test_athletes
    if overlap:
        raise AssertionError(f"Athlete-level split violated: {overlap} appear in both sets")

    return TrainTestSplit(train_df=train_df, test_df=test_df, train_athletes=train_athletes, test_athletes=test_athletes)


@dataclass
class TrainedModel:
    model: object
    feature_columns: tuple[str, ...]
    model_type: str
    train_auc: float
    test_auc: float
    test_accuracy: float
    precision: float = float("nan")
    recall: float = float("nan")
    f1_score: float = float("nan")
    pr_auc: float = float("nan")
    confusion_matrix: list = field(default_factory=list)
    n_train_athletes: int = 0
    n_test_athletes: int = 0


def train_model(
    split: TrainTestSplit,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
    model_type: str = "xgboost",
    random_state: int = 42,
) -> TrainedModel:
    """Train an interpretable ML classifier (xgboost, random_forest, or logistic_regression) on features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        auc,
    )

    # Ensure no leakage columns are in predictors
    cleaned_features = [c for c in feature_columns if c not in EXCLUDED_TARGET_COLUMNS]

    for col in (*cleaned_features, label_column):
        if col not in split.train_df.columns:
            raise ValueError(f"Missing required column '{col}' in training data")

    X_train = split.train_df[cleaned_features].fillna(0.0)
    y_train = split.train_df[label_column]
    X_test = split.test_df[cleaned_features].fillna(0.0)
    y_test = split.test_df[label_column]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training label has only one class present -- cannot train a classifier."
        )

    if model_type == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    clf.fit(X_train, y_train)

    train_pred = clf.predict_proba(X_train)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    test_class = (test_pred >= 0.5).astype(int)

    train_auc = roc_auc_score(y_train, train_pred) if y_train.nunique() > 1 else float("nan")
    test_auc = roc_auc_score(y_test, test_pred) if y_test.nunique() > 1 else float("nan")
    acc = accuracy_score(y_test, test_class)

    prec = precision_score(y_test, test_class, zero_division=0)
    rec = recall_score(y_test, test_class, zero_division=0)
    f1 = f1_score(y_test, test_class, zero_division=0)

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_test, test_pred)
        pr_auc_val = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc_val = float("nan")

    cm = confusion_matrix(y_test, test_class).tolist() if y_test.nunique() > 1 else []

    return TrainedModel(
        model=clf,
        feature_columns=tuple(cleaned_features),
        model_type=model_type,
        train_auc=float(train_auc),
        test_auc=float(test_auc),
        test_accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        pr_auc=pr_auc_val,
        confusion_matrix=cm,
        n_train_athletes=len(split.train_athletes),
        n_test_athletes=len(split.test_athletes),
    )


def save_model(trained: TrainedModel, path: str) -> None:
    import joblib
    joblib.dump({"model": trained.model, "feature_columns": trained.feature_columns, "model_type": trained.model_type}, path)


def load_model(path: str):
    import joblib
    return joblib.load(path)


def generate_synthetic_training_data(n_athletes: int = 60, events_per_athlete: int = 15, random_state: int = 42) -> pd.DataFrame:
    """Produces a DataFrame with Level B/C features to test split and model training."""
    rng = np.random.default_rng(random_state)
    rows = []
    for a in range(n_athletes):
        athlete_id = f"synthetic_ath_{a}"
        base_fppa = rng.normal(5, 2)
        injury_prob = 1 / (1 + np.exp(-(base_fppa - 8) / 2))
        for e in range(events_per_athlete):
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "fppa_left": base_fppa + rng.normal(0, 1),
                    "fppa_right": base_fppa + rng.normal(0, 1),
                    "knee_flexion_left": rng.normal(60, 10),
                    "knee_flexion_right": rng.normal(60, 10),
                    "trunk_lean": rng.normal(5, 3),
                    "peak_knee_valgus": base_fppa + rng.normal(3, 1),
                    "mean_knee_valgus": base_fppa,
                    "knee_valgus_rom": rng.normal(5, 1),
                    "knee_valgus_velocity": rng.normal(40, 10),
                    "knee_valgus_acceleration": rng.normal(200, 50),
                    "peak_knee_flexion": rng.normal(65, 5),
                    "minimum_knee_flexion": rng.normal(15, 3),
                    "knee_flexion_rom": rng.normal(50, 5),
                    "peak_hip_flexion": rng.normal(45, 5),
                    "peak_hip_abduction": rng.normal(15, 3),
                    "peak_ankle_dorsiflexion": rng.normal(25, 4),
                    "trunk_flexion": rng.normal(10, 3),
                    "trunk_lateral_flexion": rng.normal(5, 2),
                    "pelvic_obliquity": rng.normal(4, 2),
                    "knee_flexion_LSI": rng.normal(95, 5),
                    "knee_valgus_LSI": rng.normal(90, 8),
                    "knee_valgus_at_initial_contact": rng.normal(4, 1),
                    "knee_flexion_at_initial_contact": rng.normal(20, 3),
                    "valgus_duration": rng.normal(0.1, 0.02),
                    "com_displacement": rng.normal(0.05, 0.01),
                    "pose_quality": 0.95,
                    "baseline_deviation": rng.normal(1.0, 0.5),
                    "baseline_zscore": rng.normal(0.8, 0.4),
                    LABEL_COLUMN: int(rng.random() < injury_prob),
                }
            )
    return pd.DataFrame(rows)
