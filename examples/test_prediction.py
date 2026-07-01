#!/usr/bin/env python3
"""Quick test script to verify new features are working"""

import sys
sys.path.insert(0, 'src')

from interactive_predictor_numbered import predict_match, get_2026_team_features

# Test feature extraction for a known team
print("=== Testing Feature Extraction ===\n")
brazil_features = get_2026_team_features('Brazil')
print("Brazil features:")
for key, value in brazil_features.items():
    print(f"  {key}: {value:.3f}")

print("\n=== Testing Prediction ===\n")
result = predict_match('Brazil', 'Argentina')
print(f"{result['home_team']} vs {result['away_team']}")
print(f"{result['home_team']} advance: {result['home_win_prob']:.1%}")
print(f"{result['away_team']} advance: {result['away_win_prob']:.1%}")
print(f"Prediction: {result['prediction']} advances")
