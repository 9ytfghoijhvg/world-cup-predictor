#!/usr/bin/env python3
"""
Train the final Random Forest model and save it to disk.
This model will be used by the interactive predictor.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np

print("="*70)
print("TRAINING FINAL WORLD CUP KNOCKOUT PREDICTOR MODEL")
print("="*70)

# Load prepared data
df = pd.read_csv('../data/knockout_matches_prepared.csv')
print(f"\nLoaded {len(df)} knockout matches from historical data")

# Define features
feature_cols = [
    'elo_diff', 
    'home_rolling_xg_10', 
    'away_rolling_xg_10', 
    'host_advantage', 
    'home_penalty_win_rate', 
    'away_penalty_win_rate', 
    'home_group_points', 
    'home_group_gd', 
    'away_group_points', 
    'away_group_gd',
    'home_rolling_gf_10',
    'home_rolling_ga_10',
    'away_rolling_gf_10',
    'away_rolling_ga_10'
]

target_col = 'home_advanced'

# Prepare data
df_clean = df[feature_cols + [target_col]].dropna()
print(f"Clean samples after removing NaN: {len(df_clean)}")

X = df_clean[feature_cols]
y = df_clean[target_col]

# Cross-validation to check performance
print("\n" + "="*70)
print("CROSS-VALIDATION (5-fold)")
print("="*70)

model_cv = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train final model on ALL data
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*70)

model = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

print(f"\nModel configuration:")
print(f"  - Algorithm: Random Forest")
print(f"  - Number of trees: {model.n_estimators}")
print(f"  - Features: {len(feature_cols)}")
print(f"  - Training samples: {len(X)}")

model.fit(X, y)

# Feature importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.4f}")

# Save the model
model_path = '../models/random_forest_model.pkl'
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

import os
os.makedirs('../models', exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model saved to: {model_path}")

# Save model metadata
metadata = {
    'n_estimators': model.n_estimators,
    'n_features': len(feature_cols),
    'features': feature_cols,
    'training_samples': len(X),
    'cv_mean_accuracy': cv_scores.mean(),
    'cv_std_accuracy': cv_scores.std()
}

metadata_path = '../models/model_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Metadata saved to: {metadata_path}")

# Verify the saved model works
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Make a test prediction
test_prediction = loaded_model.predict_proba(X.iloc[0:1])
print(f"✅ Model loaded and tested successfully!")
print(f"   Test prediction shape: {test_prediction.shape}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nModel ready for use in interactive predictor.")
print(f"Expected accuracy: ~{cv_scores.mean():.1%}")
