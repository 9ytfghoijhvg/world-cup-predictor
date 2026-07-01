#!/usr/bin/env python3
"""Display information about the trained model"""

import pickle
import pandas as pd

print("="*70)
print("WORLD CUP PREDICTOR - MODEL INFORMATION")
print("="*70)

# Load model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('models/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"\n📊 MODEL CONFIGURATION")
print(f"   Algorithm: Random Forest Classifier")
print(f"   Number of trees: {metadata['n_estimators']}")
print(f"   Features used: {metadata['n_features']}")
print(f"   Training samples: {metadata['training_samples']}")

print(f"\n🎯 PERFORMANCE")
print(f"   Cross-validation accuracy: {metadata['cv_mean_accuracy']:.1%} (+/- {metadata['cv_std_accuracy']*2:.1%})")

print(f"\n✨ FEATURES ({len(metadata['features'])} total)")
for i, feature in enumerate(metadata['features'], 1):
    print(f"   {i:2d}. {feature}")

# Feature importance
print(f"\n🏆 FEATURE IMPORTANCE (Top 5)")
feature_importance = pd.DataFrame({
    'feature': metadata['features'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']:25s} {row['importance']:.2%}")

print("\n" + "="*70)
