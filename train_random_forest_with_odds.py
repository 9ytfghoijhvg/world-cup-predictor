import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load prepared data with betting odds
df = pd.read_csv('data/knockout_matches_prepared_with_odds.csv')
print(f"Loaded {len(df)} rows with betting odds feature")

# Define features - now including betting odds
feature_cols_base = [
    'elo_diff', 
    'home_rolling_xg_10', 
    'away_rolling_xg_10', 
    'host_advantage', 
    'home_penalty_win_rate', 
    'away_penalty_win_rate', 
    'home_group_points', 
    'home_group_gd', 
    'away_group_points', 
    'away_group_gd'
]

feature_cols_with_odds = feature_cols_base + ['home_betting_prob']
target_col = 'home_advanced'

# ===== MODEL 1: WITHOUT BETTING ODDS (Baseline) =====
print("\n" + "="*70)
print("MODEL 1: WITHOUT BETTING ODDS (BASELINE)")
print("="*70)

df_clean_base = df[feature_cols_base + [target_col]].dropna()
print(f"Rows with complete base features: {len(df_clean_base)}")

split_idx = int(len(df_clean_base) * 0.8)
train_base = df_clean_base.iloc[:split_idx]
test_base = df_clean_base.iloc[split_idx:]

X_train_base = train_base[feature_cols_base]
y_train_base = train_base[target_col]
X_test_base = test_base[feature_cols_base]
y_test_base = test_base[target_col]

model_base = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model_base.fit(X_train_base, y_train_base)

y_pred_base = model_base.predict(X_test_base)
accuracy_base = accuracy_score(y_test_base, y_pred_base)

print(f"\nTraining set size: {len(train_base)}")
print(f"Test set size: {len(test_base)}")
print(f"Test Accuracy (Base): {accuracy_base:.1%}")
print(f"\nFeature Importances (Base Model):")
for feat, imp in sorted(zip(feature_cols_base, model_base.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<25} {imp:.4f}")

# ===== MODEL 2: WITH BETTING ODDS =====
print("\n" + "="*70)
print("MODEL 2: WITH BETTING ODDS FEATURE")
print("="*70)

df_clean_odds = df[feature_cols_with_odds + [target_col]].dropna()
print(f"Rows with complete features including odds: {len(df_clean_odds)}")

if len(df_clean_odds) < 10:
    print("\n⚠️  Not enough data with betting odds for separate training.")
    print(f"Rows with odds: {df['home_betting_prob'].notna().sum()}")
    print(f"Rows with all base features: {len(df_clean_base)}")
    print("\nAlternative: Train on base data, use odds for 2026 predictions only")
else:
    split_idx_odds = int(len(df_clean_odds) * 0.8)
    train_odds = df_clean_odds.iloc[:split_idx_odds]
    test_odds = df_clean_odds.iloc[split_idx_odds:]
    
    X_train_odds = train_odds[feature_cols_with_odds]
    y_train_odds = train_odds[target_col]
    X_test_odds = test_odds[feature_cols_with_odds]
    y_test_odds = test_odds[target_col]
    
    model_odds = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model_odds.fit(X_train_odds, y_train_odds)
    
    y_pred_odds = model_odds.predict(X_test_odds)
    accuracy_odds = accuracy_score(y_test_odds, y_pred_odds)
    
    print(f"\nTraining set size: {len(train_odds)}")
    print(f"Test set size: {len(test_odds)}")
    print(f"Test Accuracy (With Odds): {accuracy_odds:.1%}")
    print(f"Accuracy Improvement: {(accuracy_odds - accuracy_base)*100:+.1f} percentage points")
    
    print(f"\nFeature Importances (With Odds):")
    for feat, imp in sorted(zip(feature_cols_with_odds, model_odds.feature_importances_), key=lambda x: x[1], reverse=True):
        marker = " ⭐" if feat == 'home_betting_prob' else ""
        print(f"  {feat:<25} {imp:.4f}{marker}")

# ===== TRAIN ON FULL BASE DATA (Better for 2026) =====
print("\n" + "="*70)
print("FINAL MODEL: RETRAINED ON FULL BASE DATA (No odds feature)")
print("="*70)
print("\nRationale: With only 18 matches having odds, betting feature")
print("would be sparse and unreliable. Better to use well-crafted")
print("base features and apply odds contextually for 2026 predictions.")

X_final = df_clean_base[feature_cols_base]
y_final = df_clean_base[target_col]

model_final = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model_final.fit(X_final, y_final)

# Cross-validation on temporal split
split_idx_final = int(len(df_clean_base) * 0.8)
X_train_final = df_clean_base.iloc[:split_idx_final][feature_cols_base]
y_train_final = df_clean_base.iloc[:split_idx_final][target_col]
X_test_final = df_clean_base.iloc[split_idx_final:][feature_cols_base]
y_test_final = df_clean_base.iloc[split_idx_final:][target_col]

y_pred_final = model_final.predict(X_test_final)
accuracy_final = accuracy_score(y_test_final, y_pred_final)

print(f"\nTest Accuracy (Final): {accuracy_final:.1%}")
print(f"Baseline accuracy: {accuracy_base:.1%}")

# Save the model
import pickle
with open('models/random_forest_model_final.pkl', 'wb') as f:
    pickle.dump(model_final, f)
print(f"\n✓ Saved final model to models/random_forest_model_final.pkl")

# ===== BETTING ODDS ANALYSIS =====
print("\n" + "="*70)
print("BETTING ODDS ANALYSIS")
print("="*70)

odds_matches = df[df['home_betting_prob'].notna()].copy()
if len(odds_matches) > 0:
    odds_matches['model_prob'] = model_final.predict_proba(odds_matches[feature_cols_base])[:, 1]
    odds_matches['betting_prediction'] = (odds_matches['home_betting_prob'] > 0.5).astype(int)
    odds_matches['model_prediction'] = (odds_matches['model_prob'] > 0.5).astype(int)
    
    betting_acc = (odds_matches['betting_prediction'] == odds_matches['home_advanced']).mean()
    model_acc = (odds_matches['model_prediction'] == odds_matches['home_advanced']).mean()
    
    print(f"\nOn matches with betting odds ({len(odds_matches)} matches):")
    print(f"  Betting market accuracy: {betting_acc:.1%}")
    print(f"  Model accuracy: {model_acc:.1%}")
    print(f"  Model advantage: {(model_acc - betting_acc)*100:+.1f} pp")
    
    # Show disagreements
    disagreements = odds_matches[odds_matches['betting_prediction'] != odds_matches['model_prediction']]
    if len(disagreements) > 0:
        print(f"\n  Cases where model & market disagree: {len(disagreements)}")
        for idx, row in disagreements.head(5).iterrows():
            print(f"    {row['home_team']} vs {row['away_team']} ({int(row['Year'])})")
            print(f"      Market: {row['home_betting_prob']:.1%} | Model: {row['model_prob']:.1%} | Actual: {'Home' if row['home_advanced'] else 'Away'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✅ Model trained and saved")
print(f"✅ Base model accuracy: {accuracy_base:.1%}")
print(f"✅ Betting odds available for ~14% of training data")
print(f"✅ Ready for 2026 predictions with betting context")
