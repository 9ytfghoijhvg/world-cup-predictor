#!/usr/bin/env python3
"""
Log all Quarterfinal predictions and actual results for tracking model performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction_tracker import log_prediction, display_accuracy, show_predictions_table
from interactive_predictor_numbered import predict_match

# Quarterfinal matches and their results (July 9-11, 2026)
# ALL 4 MATCHES COMPLETED
qf_matches = [
    # July 9
    {
        'home': 'France',
        'away': 'Morocco',
        'home_score': 2,
        'away_score': 0,
        'winner': 'France',
        'date': '2026-07-09'
    },
    # July 10
    {
        'home': 'Spain',
        'away': 'Belgium',
        'home_score': 2,
        'away_score': 1,
        'winner': 'Spain',
        'date': '2026-07-10'
    },
    # July 11
    {
        'home': 'Norway',
        'away': 'England',
        'home_score': 1,
        'away_score': 2,
        'winner': 'England',
        'date': '2026-07-11',
        'notes': 'Extra time'
    },
    {
        'home': 'Argentina',
        'away': 'Switzerland',
        'home_score': 3,
        'away_score': 1,
        'winner': 'Argentina',
        'date': '2026-07-11',
        'notes': 'Extra time'
    },
]

print("\n" + "="*100)
print("LOGGING QUARTERFINAL PREDICTIONS AND RESULTS")
print("="*100)

# Log each match
for i, match in enumerate(qf_matches, 1):
    home = match['home']
    away = match['away']
    actual_home_score = match['home_score']
    actual_away_score = match['away_score']
    actual_winner = match['winner']
    
    print(f"\n({i}) {home} vs {away}...", end=' ', flush=True)
    
    try:
        # Get prediction
        result = predict_match(home, away)
        predicted_winner = result.get('prediction', home)
        home_win_prob = result.get('home_win_prob', 0.5)
        
        # Get score prediction
        if result.get('predicted_score'):
            predicted_home_score = result['predicted_score']['home_goals_rounded']
            predicted_away_score = result['predicted_score']['away_goals_rounded']
        else:
            predicted_home_score = 0
            predicted_away_score = 0
        
        # Log prediction with actual result  
        log_row = log_prediction(
            home_team=home,
            away_team=away,
            predicted_winner=predicted_winner,
            home_win_prob=home_win_prob,
            predicted_home_score=predicted_home_score,
            predicted_away_score=predicted_away_score,
            actual_winner=actual_winner,
            actual_home_score=actual_home_score,
            actual_away_score=actual_away_score
        )
        
        # Show result
        prediction_correct = "✅" if predicted_winner == actual_winner else "❌"
        print(f"{prediction_correct} Predicted: {predicted_winner} | Actual: {actual_winner}")
        print(f"    Score: Predicted {predicted_home_score}-{predicted_away_score} | Actual {actual_home_score}-{actual_away_score}")
        
    except Exception as e:
        print(f"Error processing {home} vs {away}: {str(e)}")

# Show summary statistics
print("\n")
show_predictions_table()
display_accuracy()

print("✅ QF Predictions logged to data/prediction_log.csv")
