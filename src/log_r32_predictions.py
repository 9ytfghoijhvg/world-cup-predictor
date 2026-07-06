#!/usr/bin/env python3
"""
Log all Round of 32 predictions and actual results for tracking model performance.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction_tracker import log_prediction, display_accuracy, show_predictions_table
from interactive_predictor_numbered import predict_match

# Load model
MODEL_PATH = 'models/random_forest_model.pkl'
SCORE_MODEL_HOME_PATH = 'models/score_predictor_home.pkl'
SCORE_MODEL_AWAY_PATH = 'models/score_predictor_away.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCORE_MODEL_HOME_PATH, 'rb') as f:
        score_model_home = pickle.load(f)
    with open(SCORE_MODEL_AWAY_PATH, 'rb') as f:
        score_model_away = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Round of 32 matches and their results (as of July 2, 2026)
r32_matches = [
    # June 28
    {
        'home': 'Canada',
        'away': 'South Africa',
        'home_score': 1,
        'away_score': 0,
        'winner': 'Canada'
    },
    # June 29
    {
        'home': 'Germany',
        'away': 'Paraguay',
        'home_score': 1,
        'away_score': 1,
        'winner': 'Paraguay',  # Won penalties
        'notes': 'Penalties 4-3'
    },
    {
        'home': 'Netherlands',
        'away': 'Morocco',
        'home_score': 1,
        'away_score': 1,
        'winner': 'Morocco',  # Won penalties
        'notes': 'Penalties 3-2'
    },
    {
        'home': 'Brazil',
        'away': 'Japan',
        'home_score': 2,
        'away_score': 1,
        'winner': 'Brazil'
    },
    # June 30
    {
        'home': 'France',
        'away': 'Sweden',
        'home_score': 3,
        'away_score': 0,
        'winner': 'France'
    },
    {
        'home': 'Norway',
        'away': 'Ivory Coast',
        'home_score': 2,
        'away_score': 1,
        'winner': 'Norway'
    },
    {
        'home': 'Mexico',
        'away': 'Ecuador',
        'home_score': 2,
        'away_score': 0,
        'winner': 'Mexico'
    },
    # July 1
    {
        'home': 'England',
        'away': 'DR Congo',
        'home_score': 2,
        'away_score': 1,
        'winner': 'England'
    },
    {
        'home': 'United States',
        'away': 'Bosnia and Herzegovina',
        'home_score': 2,
        'away_score': 0,
        'winner': 'United States'
    },
    {
        'home': 'Belgium',
        'away': 'Senegal',
        'home_score': 3,
        'away_score': 2,
        'winner': 'Belgium',
        'notes': 'AET - Belgium came back from 2-0 down'
    },
]

print("\n" + "="*100)
print("LOGGING ROUND OF 32 PREDICTIONS AND RESULTS")
print("="*100)

# Log each match
for i, match in enumerate(r32_matches, 1):
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

print("✅ Predictions logged to data/prediction_log.csv")
