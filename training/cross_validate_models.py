import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load prepared data
df = pd.read_csv('data/knockout_matches_prepared.csv')
print(f"Loaded {len(df)} rows\n")

# OLD FEATURES (10 features)
old_features = ['elo_diff', 
'home_rolling_xg_10', 
'away_rolling_xg_10', 
'host_advantage', 
'home_penalty_win_rate', 
'away_penalty_win_rate', 
'home_group_points', 
'home_group_gd', 
'away_group_points', 
'away_group_gd']

# NEW FEATURES (14 features - adds rolling goals)
new_features = old_features + [
'home_rolling_gf_10',
'home_rolling_ga_10',
'away_rolling_gf_10',
'away_rolling_ga_10']

target_col = 'home_advanced'

print("="*70)
print("CROSS-VALIDATION COMPARISON (5-fold)")
print("="*70)

# OLD MODEL
df_clean_old = df[old_features + [target_col]].dropna()
X_old = df_clean_old[old_features]
y_old = df_clean_old[target_col]

model_old = RandomForestClassifier(n_estimators=1000, random_state=50)
scores_old = cross_val_score(model_old, X_old, y_old, cv=5, scoring='accuracy')

print(f"\nOLD MODEL (10 features, no rolling goals):")
print(f"  Samples: {len(df_clean_old)}")
print(f"  Cross-validation scores: {[f'{s:.3f}' for s in scores_old]}")
print(f"  Mean accuracy: {scores_old.mean():.3f} (+/- {scores_old.std() * 2:.3f})")

# NEW MODEL
df_clean_new = df[new_features + [target_col]].dropna()
X_new = df_clean_new[new_features]
y_new = df_clean_new[target_col]

model_new = RandomForestClassifier(n_estimators=1000, random_state=50)
scores_new = cross_val_score(model_new, X_new, y_new, cv=5, scoring='accuracy')

print(f"\nNEW MODEL (14 features, with rolling goals):")
print(f"  Samples: {len(df_clean_new)}")
print(f"  Cross-validation scores: {[f'{s:.3f}' for s in scores_new]}")
print(f"  Mean accuracy: {scores_new.mean():.3f} (+/- {scores_new.std() * 2:.3f})")

# Comparison
print("\n" + "="*70)
print("RESULTS")
print("="*70)
improvement = (scores_new.mean() - scores_old.mean()) * 100
print(f"Old model: {scores_old.mean():.3f}")
print(f"New model: {scores_new.mean():.3f}")
print(f"Improvement: {improvement:+.1f} percentage points")

if improvement > 0:
    print(f"\n✅ New model is better!")
elif improvement < 0:
    print(f"\n⚠️  Old model performed better")
else:
    print(f"\n➡️  Models perform equally")

# Train final model on all data
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*70)
model_final = RandomForestClassifier(n_estimators=1000, random_state=50)
model_final.fit(X_new, y_new)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': new_features,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features by Importance:")
print(feature_importance.head(10).to_string(index=False))

# Highlight rolling goals features
print("\nRolling Goals Features Ranking:")
rolling_features = feature_importance[feature_importance['feature'].str.contains('rolling_g')]
for idx, row in rolling_features.iterrows():
    rank = feature_importance.index.get_loc(idx) + 1
    print(f"  #{rank:2d} - {row['feature']:25s} {row['importance']:.4f}")
