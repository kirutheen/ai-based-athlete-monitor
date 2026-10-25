"""
Tests for src/models/train_risk_model.py.

These tests verify the ML PLUMBING is correct (athlete-disjoint split,
training runs, evaluation metrics compute) using
generate_synthetic_training_data(), which is clearly documented as
placeholder data. They do NOT and cannot validate real injury
prediction -- see the module docstring. A passing test here means "the
training pipeline is ready for real data," not "the model works."
"""

from __future__ import annotations

from src.models.train_risk_model import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    athlete_level_split,
    generate_synthetic_training_data,
    load_model,
    save_model,
    train_model,
)


def test_athlete_level_split_has_no_overlap():
    df = generate_synthetic_training_data(n_athletes=40, events_per_athlete=10)
    split = athlete_level_split(df, test_size=0.25)
    assert split.train_athletes.isdisjoint(split.test_athletes)
    assert len(split.train_athletes) + len(split.test_athletes) == 40


def test_athlete_level_split_rejects_missing_athlete_id():
    df = generate_synthetic_training_data(n_athletes=5).drop(columns=["athlete_id"])
    try:
        athlete_level_split(df)
        assert False, "expected ValueError for missing athlete_id column"
    except ValueError:
        pass


def test_train_model_runs_and_evaluates():
    df = generate_synthetic_training_data(n_athletes=60, events_per_athlete=15)
    split = athlete_level_split(df, test_size=0.2)
    trained = train_model(split, feature_columns=FEATURE_COLUMNS, label_column=LABEL_COLUMN)

    assert 0.0 <= trained.test_accuracy <= 1.0
    assert trained.n_train_athletes + trained.n_test_athletes == 60
    # AUC should be computable (both classes present given the synthetic
    # label design); not asserting it's "good", since this is synthetic
    # noise, not a real validated prediction target.
    assert trained.train_auc == trained.train_auc  # not NaN
    assert trained.test_auc == trained.test_auc


def test_train_model_rejects_missing_feature_column():
    df = generate_synthetic_training_data(n_athletes=20, events_per_athlete=5)
    split = athlete_level_split(df)
    try:
        train_model(split, feature_columns=("nonexistent_feature",), label_column=LABEL_COLUMN)
        assert False, "expected ValueError for missing feature column"
    except ValueError:
        pass


def test_train_model_rejects_single_class_label():
    df = generate_synthetic_training_data(n_athletes=20, events_per_athlete=5)
    df[LABEL_COLUMN] = 0  # force a single class
    split = athlete_level_split(df)
    try:
        train_model(split, feature_columns=FEATURE_COLUMNS, label_column=LABEL_COLUMN)
        assert False, "expected ValueError for single-class label"
    except ValueError:
        pass


def test_save_and_load_model_roundtrip(tmp_path):
    df = generate_synthetic_training_data(n_athletes=60, events_per_athlete=15)
    split = athlete_level_split(df, test_size=0.2)
    trained = train_model(split, feature_columns=FEATURE_COLUMNS, label_column=LABEL_COLUMN)

    model_path = tmp_path / "model.joblib"
    save_model(trained, str(model_path))
    assert model_path.exists()

    loaded = load_model(str(model_path))
    assert loaded["feature_columns"] == FEATURE_COLUMNS
    # The loaded model should produce the same predictions as the original.
    X_test = split.test_df[list(FEATURE_COLUMNS)].fillna(0.0)
    original_preds = trained.model.predict_proba(X_test)[:, 1]
    loaded_preds = loaded["model"].predict_proba(X_test)[:, 1]
    assert (original_preds == loaded_preds).all()
