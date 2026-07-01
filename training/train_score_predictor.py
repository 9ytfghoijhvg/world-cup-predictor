#!/usr/bin/env python3
"""
Train a Poisson regression model to predict match scores.

Poisson regression is ideal for predicting count data (goals scored).
We'll train two models:
1. Home goals model
2. Away goals model
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

print("="*70)
print("TRAINING SCORE PREDICTION MODEL (POISSON REGRESSION)")
print("="*70)

# Load prepared data
df = pd.read_csv('../data/knockout_matches_prepared.csv')
print(f"\nLoaded {len(df)} knockout matches")

# Features for score prediction
score_features = [
    'elo_diff',
    'home_rolling_xg_10',
    'away_rolling_xg_10',
    'home_rolling_gf_10',
    'home_rolling_ga_10',
    'away_rolling_gf_10',
    'away_rolling_ga_10',
    'host_advantage',
]

# Targets
home_target = 'home_score'
away_target = 'away_score'

# Clean data
df_clean = df[score_features + [home_target, away_target]].dropna()
print(f"Clean samples: {len(df_clean)}")

X = df_clean[score_features]
y_home = df_clean[home_target]
y_away = df_clean[away_target]

# Train home goals model
print("\n" + "="*70)
print("TRAINING HOME GOALS MODEL")
print("="*70)

model_home = PoissonRegressor(alpha=0.1, max_iter=1000)
model_home.fit(X, y_home)

# Cross-validation
cv_scores_home = cross_val_score(model_home, X, y_home, cv=5, 
                                  scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores_home.mean():.3f} (+/- {cv_scores_home.std() * 2:.3f})")

# Predict and evaluate
y_home_pred = model_home.predict(X)
mae_home = mean_absolute_error(y_home, y_home_pred)
rmse_home = np.sqrt(mean_squared_error(y_home, y_home_pred))

print(f"Training MAE: {mae_home:.3f} goals")
print(f"Training RMSE: {rmse_home:.3f} goals")

# Train away goals model
print("\n" + "="*70)
print("TRAINING AWAY GOALS MODEL")
print("="*70)

model_away = PoissonRegressor(alpha=0.1, max_iter=1000)
model_away.fit(X, y_away)

# Cross-validation
cv_scores_away = cross_val_score(model_away, X, y_away, cv=5, 
                                  scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores_away.mean():.3f} (+/- {cv_scores_away.std() * 2:.3f})")

# Predict and evaluate
y_away_pred = model_away.predict(X)
mae_away = mean_absolute_error(y_away, y_away_pred)
rmse_away = np.sqrt(mean_squared_error(y_away, y_away_pred))

print(f"Training MAE: {mae_away:.3f} goals")
print(f"Training RMSE: {rmse_away:.3f} goals")

# Feature importance (coefficients)
print("\n" + "="*70)
print("FEATURE COEFFICIENTS")
print("="*70)

coef_df = pd.DataFrame({
    'feature': score_features,
    'home_goals_coef': model_home.coef_,
    'away_goals_coef': model_away.coef_
})

print("\nHome Goals Model:")
for idx, row in coef_df.nlargest(5, 'home_goals_coef').iterrows():
    print(f"  {row['feature']:25s} {row['home_goals_coef']:+.4f}")

print("\nAway Goals Model:")
for idx, row in coef_df.nlargest(5, 'away_goals_coef').iterrows():
    print(f"  {row['feature']:25s} {row['away_goals_coef']:+.4f}")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

import os
os.makedirs('../models', exist_ok=True)

# Save home goals model
with open('../models/score_predictor_home.pkl', 'wb') as f:
    pickle.dump(model_home, f)
print("✅ Saved home goals model")

# Save away goals model
with open('../models/score_predictor_away.pkl', 'wb') as f:
    pickle.dump(model_away, f)
print("✅ Saved away goals model")

# Save metadata
metadata = {
    'features': score_features,
    'home_mae': mae_home,
    'away_mae': mae_away,
    'home_cv_mae': -cv_scores_home.mean(),
    'away_cv_mae': -cv_scores_away.mean(),
    'training_samples': len(df_clean)
}

with open('../models/score_predictor_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Saved metadata")

# Test predictions
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Show some example predictions
for idx in [0, 10, 20, 30, 40]:
    if idx < len(df_clean):
        row = df_clean.iloc[idx]
        X_test = row[score_features].values.reshape(1, -1)
        
        pred_home = model_home.predict(X_test)[0]
        pred_away = model_away.predict(X_test)[0]
        actual_home = row[home_target]
        actual_away = row[away_target]
        
        print(f"\nPredicted: {pred_home:.1f} - {pred_away:.1f}")
        print(f"Actual:    {actual_home:.0f} - {actual_away:.0f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nExpected prediction error: ~{(mae_home + mae_away)/2:.2f} goals per team")
