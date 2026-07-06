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

# NEWEST FEATURES (21 features - adds h2h and betting odds)
newest_features = new_features + [
'home_h2h_win_rate',
'away_h2h_win_rate',
'home_h2h_goal_diff',
'away_h2h_goal_diff',
'h2h_meeting_count',
'home_odds_prob',
'away_odds_prob']

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

# NEWEST MODEL
df_clean_newest = df[newest_features + [target_col]].dropna()
X_newest = df_clean_newest[newest_features]
y_newest = df_clean_newest[target_col]

model_newest = RandomForestClassifier(n_estimators=1000, random_state=50)
scores_newest = cross_val_score(model_newest, X_newest, y_newest, cv=5, scoring='accuracy')

print(f"\nNEWEST MODEL (21 features, adds h2h + betting odds):")
print(f"  Samples: {len(df_clean_newest)}")
print(f"  Cross-validation scores: {[f'{s:.3f}' for s in scores_newest]}")
print(f"  Mean accuracy: {scores_newest.mean():.3f} (+/- {scores_newest.std() * 2:.3f})")

# Comparison
print("\n" + "="*70)
print("RESULTS")
print("="*70)
improvement_v2 = (scores_new.mean() - scores_old.mean()) * 100
improvement_v3 = (scores_newest.mean() - scores_new.mean()) * 100
print(f"Old model (10 features): {scores_old.mean():.3f}")
print(f"New model (14 features): {scores_new.mean():.3f} ({improvement_v2:+.1f} pp)")
print(f"Newest model (21 features): {scores_newest.mean():.3f} ({improvement_v3:+.1f} pp)")

if scores_newest.mean() > scores_new.mean():
    print(f"\n✅ Newest model with h2h and odds is best!")
elif scores_new.mean() > scores_old.mean():
    print(f"\n✅ Model with rolling goals is best!")
else:
    print(f"\n⚠️  Original model still best")

# Train final model on all data (use best features)
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*70)
model_final = RandomForestClassifier(n_estimators=1000, random_state=50)
model_final.fit(X_newest, y_newest)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': newest_features,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features by Importance:")
print(feature_importance.head(15).to_string(index=False))

# Highlight rolling goals features
print("\nRolling Goals Features Ranking:")
rolling_features = feature_importance[feature_importance['feature'].str.contains('rolling_g')]
for idx, row in rolling_features.iterrows():
    rank = feature_importance.index.get_loc(idx) + 1
    print(f"  #{rank:2d} - {row['feature']:25s} {row['importance']:.4f}")
