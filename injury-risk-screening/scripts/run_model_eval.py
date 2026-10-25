"""
Model evaluation script for DriftGuard.
Generates 900 samples (60 subjects x 15 trials), runs GroupShuffleSplit,
trains XGBoost and Random Forest risk models, and exports data/raw/movement_features.csv.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.train_risk_model import (
    generate_synthetic_training_data,
    athlete_level_split,
    train_model,
)


if __name__ == "__main__":
    print("Generating 900 samples across 60 subjects (15 trials per subject)...")
    df = generate_synthetic_training_data(n_athletes=60, events_per_athlete=15, random_state=42)

    # Save to data/raw/movement_features.csv
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/movement_features.csv", index=False)
    print(f"Saved {len(df)} records to data/raw/movement_features.csv")

    # Perform athlete-disjoint split (48 train athletes, 12 test athletes)
    split = athlete_level_split(df, test_size=0.2, random_state=42)

    # Train XGBoost
    xgb_model = train_model(split, model_type="xgboost")
    print("\n--- XGBoost Model Evaluation Results ---")
    print(f"Train Athletes: {xgb_model.n_train_athletes} | Test Athletes: {xgb_model.n_test_athletes}")
    print(f"Train AUC: {xgb_model.train_auc:.4f} | Test AUC: {xgb_model.test_auc:.4f}")
    print(f"Test Accuracy: {xgb_model.test_accuracy*100:.1f}%")
    print(f"Precision: {xgb_model.precision*100:.1f}% | Recall: {xgb_model.recall*100:.1f}% | F1-Score: {xgb_model.f1_score*100:.1f}%")
    print(f"Confusion Matrix [TN, FP / FN, TP]: {xgb_model.confusion_matrix}")

    # Train Random Forest
    rf_model = train_model(split, model_type="random_forest")
    print("\n--- Random Forest Model Evaluation Results ---")
    print(f"Train AUC: {rf_model.train_auc:.4f} | Test AUC: {rf_model.test_auc:.4f}")
    print(f"Test Accuracy: {rf_model.test_accuracy*100:.1f}%")
    print(f"Confusion Matrix [TN, FP / FN, TP]: {rf_model.confusion_matrix}")
