#!/usr/bin/env python3
"""Test multiple predictions to showcase the improved model"""

import sys
sys.path.insert(0, 'src')

from interactive_predictor_numbered import predict_match

print("="*70)
print("2026 WORLD CUP PREDICTOR - SAMPLE PREDICTIONS")
print("="*70)

# Test interesting matchups
matchups = [
    ('Brazil', 'Argentina', 'Classic South American rivalry'),
    ('England', 'Germany', 'Historic European clash'),
    ('Spain', 'France', 'Recent powerhouses'),
    ('United States', 'Mexico', 'CONCACAF rivals'),
    ('Argentina', 'France', '2022 Final rematch'),
]

for home, away, description in matchups:
    print(f"\n{description}")
    print(f"{'-'*70}")
    result = predict_match(home, away)
    
    print(f"  {result['home_team']:20s} vs {result['away_team']:20s}")
    
    # Show predicted score if available
    if result.get('predicted_score'):
        score = result['predicted_score']
        print(f"  📊 Predicted Score: {score['home_goals_rounded']}-{score['away_goals_rounded']} (Expected: {score['home_goals']:.1f}-{score['away_goals']:.1f})")
    
    print(f"  {result['home_team']:20s} advance: {result['home_win_prob']:>6.1%}")
    print(f"  {result['away_team']:20s} advance: {result['away_win_prob']:>6.1%}")
    print(f"  🏆 Prediction: {result['prediction']} advances")

print("\n" + "="*70)
print("Model uses 552 training samples with 14 features")
print("Expected accuracy: ~65% (based on cross-validation)")
print("="*70)
