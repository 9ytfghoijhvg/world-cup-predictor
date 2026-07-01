import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Train with OLD features
print("="*70)
print("MODEL 1: OLD FEATURES (10 features - no rolling goals)")
print("="*70)
df_clean_old = df[old_features + [target_col]].dropna()
split_idx = int(len(df_clean_old) * 0.8)
train_old = df_clean_old.iloc[:split_idx]
test_old = df_clean_old.iloc[split_idx:]

X_train_old = train_old[old_features]
y_train_old = train_old['home_advanced']
X_test_old = test_old[old_features]
y_test_old = test_old['home_advanced']

model_old = RandomForestClassifier(n_estimators=10000, random_state=50)
model_old.fit(X_train_old, y_train_old)
y_pred_old = model_old.predict(X_test_old)
accuracy_old = accuracy_score(y_test_old, y_pred_old)

print(f"Training samples: {len(train_old)}")
print(f"Test samples: {len(test_old)}")
print(f"Test Accuracy: {accuracy_old:.3f}")
print(f"Correct predictions: {sum(y_pred_old == y_test_old)}/{len(y_test_old)}")

# Train with NEW features
print("\n" + "="*70)
print("MODEL 2: NEW FEATURES (14 features - with rolling goals)")
print("="*70)
df_clean_new = df[new_features + [target_col]].dropna()
split_idx = int(len(df_clean_new) * 0.8)
train_new = df_clean_new.iloc[:split_idx]
test_new = df_clean_new.iloc[split_idx:]

X_train_new = train_new[new_features]
y_train_new = train_new['home_advanced']
X_test_new = test_new[new_features]
y_test_new = test_new['home_advanced']

model_new = RandomForestClassifier(n_estimators=10000, random_state=50)
model_new.fit(X_train_new, y_train_new)
y_pred_new = model_new.predict(X_test_new)
accuracy_new = accuracy_score(y_test_new, y_pred_new)

print(f"Training samples: {len(train_new)}")
print(f"Test samples: {len(test_new)}")
print(f"Test Accuracy: {accuracy_new:.3f}")
print(f"Correct predictions: {sum(y_pred_new == y_test_new)}/{len(y_test_new)}")

# Show improvement
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Old Model Accuracy: {accuracy_old:.3f}")
print(f"New Model Accuracy: {accuracy_new:.3f}")
improvement = (accuracy_new - accuracy_old) * 100
print(f"Improvement: {improvement:+.1f} percentage points")

# Feature importance for new features only
print("\n" + "="*70)
print("NEW ROLLING GOALS FEATURES - Importance Ranking")
print("="*70)
feature_importance = pd.DataFrame({
    'feature': new_features,
    'importance': model_new.feature_importances_
}).sort_values('importance', ascending=False)

rolling_goals_features = feature_importance[feature_importance['feature'].str.contains('rolling_g')]
print(rolling_goals_features.to_string(index=False))

print("\n" + "="*70)
print("TOP 5 MOST IMPORTANT FEATURES (NEW MODEL)")
print("="*70)
print(feature_importance.head(5).to_string(index=False))
