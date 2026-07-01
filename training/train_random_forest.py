import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load prepared data
df = pd.read_csv('data/knockout_matches_prepared.csv')
print(f"Loaded {len(df)} rows")

feature_cols = ['elo_diff', 
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
'away_rolling_ga_10']

target_col = 'home_advanced'

df_clean = df[feature_cols + [target_col]].dropna()
split_idx = int(len(df_clean) * 0.8)  # 80% for training
train = df_clean.iloc[:split_idx]
test = df_clean.iloc[split_idx:]

X_train = train[feature_cols]
y_train = train['home_advanced']
X_test = test[feature_cols]
y_test = test['home_advanced']

model = RandomForestClassifier(n_estimators=10000, random_state=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"Random Forest - Test Accuracy: {accuracy:.3f}")
print(f"{'='*60}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Wins', 'Home Wins']))

# Feature Importance
print("\nFeature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print(f"\n{'='*60}")
print(f"Training complete! Model has {len(feature_cols)} features.")
print(f"{'='*60}")
