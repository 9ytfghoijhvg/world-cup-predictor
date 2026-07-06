#!/usr/bin/env python3
"""
Log all Round of 16 predictions and actual results for tracking model performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction_tracker import log_prediction, display_accuracy, show_predictions_table
from interactive_predictor_numbered import predict_match

# Round of 16 matches and their results (July 4-7, 2026)
# NOTE: Only includes matches that have been completed so far
r16_matches = [
    # July 4 - COMPLETED
    {
        'home': 'Paraguay',
        'away': 'France',
        'home_score': 0,
        'away_score': 1,
        'winner': 'France',
        'date': '2026-07-04'
    },
    {
        'home': 'Canada',
        'away': 'Morocco',
        'home_score': 0,
        'away_score': 3,
        'winner': 'Morocco',
        'date': '2026-07-04'
    },
    # July 5 - COMPLETED
    {
        'home': 'Brazil',
        'away': 'Norway',
        'home_score': 1,
        'away_score': 2,
        'winner': 'Norway',
        'date': '2026-07-05'
    },
    # UPCOMING (not logged yet):
    # July 5 - Mexico vs England
    # July 6 - USA vs Belgium, Spain vs Portugal
    # July 7 - Argentina vs Egypt, Colombia vs Switzerland
]

print("\n" + "="*100)
print("LOGGING ROUND OF 16 PREDICTIONS AND RESULTS")
print("="*100)

# Log each match
for i, match in enumerate(r16_matches, 1):
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

print("✅ R16 Predictions logged to data/prediction_log.csv")
