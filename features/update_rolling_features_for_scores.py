"""
Automatic rolling feature update system for SCORE prediction.

This module provides functions to:
1. Add played tournament matches to international_results.csv
2. Recalculate rolling features with tournament data included
3. Update score predictions with fresh rolling stats

This ensures score predictions reflect current tournament goal-scoring form,
not stagnant pre-tournament statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def get_updated_rolling_goals_for_scores(
    international_csv_path='data/international_results.csv',
    window=10
):
    """
    Calculate rolling goals features specifically for score prediction.
    
    Returns:
    - DataFrame with Date, team, rolling_gf (goals for), rolling_ga (goals against)
    """
    from .rolling_goals_feature import get_rolling_goals
    
    # Load updated international results
    international = pd.read_csv(international_csv_path)
    international['Date'] = pd.to_datetime(international['date']).dt.strftime('%Y-%m-%d') if 'date' in international.columns else international['Date']
    
    # Calculate rolling goals with updated data
    rolling_goals = get_rolling_goals(international, window=window)
    
    print(f"✓ Recalculated rolling goals with {len(international)} total matches")
    
    return rolling_goals


def prepare_score_prediction_data_with_updated_rolling(
    prediction_log_path='data/prediction_log.csv',
    international_csv_path='data/international_results.csv',
    output_path='data/score_prediction_data_updated.csv',
    window=10
):
    """
    Prepare score prediction data with updated rolling features.
    
    This pipeline:
    1. Uses updated international_results.csv (includes tournament matches)
    2. Recalculates rolling goals features
    3. Prepares data for score prediction with fresh rolling stats
    
    Returns:
    - DataFrame with updated score prediction features
    """
    from .rolling_goals_feature import get_rolling_goals
    from .update_rolling_features import update_international_results, load_prediction_log
    
    print(f"\n--- Preparing score prediction data with updated rolling features ---")
    
    # Step 1: Update international results with played matches
    new_matches, count_added = update_international_results(
        international_csv_path=international_csv_path,
        prediction_log_path=prediction_log_path,
        backup=True
    )
    
    # Step 2: Load updated international results
    international = pd.read_csv(international_csv_path)
    international['Date'] = pd.to_datetime(international['date']).dt.strftime('%Y-%m-%d') if 'date' in international.columns else international['Date']
    
    # Step 3: Calculate rolling goals
    rolling_goals = get_rolling_goals(international, window=window)
    
    # Step 4: Load historical knockout matches for score training
    historical_ko = pd.read_csv('data/matches_1930_2022.csv')
    
    # Step 5: Get knockout rounds
    knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                       'Quarter-finals', 'Round of 16', 'Second round',
                       'Final stage']
    
    ko = historical_ko[historical_ko['Round'].isin(knockout_rounds)].copy()
    
    # Step 6: Prepare data for home team perspective
    ko_home = ko[['Date', 'home_team', 'home_score', 'away_score']].copy()
    ko_home.columns = ['Date', 'team', 'goals_scored', 'goals_conceded']
    
    # Step 7: Prepare data for away team perspective
    ko_away = ko[['Date', 'away_team', 'away_score', 'home_score']].copy()
    ko_away.columns = ['Date', 'team', 'goals_scored', 'goals_conceded']
    
    # Step 8: Combine both perspectives
    ko_combined = pd.concat([ko_home, ko_away], ignore_index=True)
    
    # Step 9: Merge rolling goals features
    ko_combined = ko_combined.merge(
        rolling_goals[['Date', 'team', 'rolling_gf', 'rolling_ga']],
        left_on=['Date', 'team'],
        right_on=['Date', 'team'],
        how='left'
    )
    
    # Step 10: Save prepared data
    ko_combined.to_csv(output_path, index=False)
    print(f"✓ Saved updated score prediction data: {output_path}")
    print(f"  Total rows: {len(ko_combined)}")
    
    return ko_combined


def get_team_rolling_stats_for_score_prediction(
    team_name,
    international_csv_path='data/international_results.csv',
    window=10
):
    """
    Get updated rolling goal stats for a team (for score prediction).
    
    Parameters:
    - team_name: Team to get stats for
    - international_csv_path: Path to updated international_results.csv
    - window: Rolling window size
    
    Returns:
    - dict with rolling_gf (goals for), rolling_ga (goals against)
    """
    from .rolling_goals_feature import get_rolling_goals
    
    international = pd.read_csv(international_csv_path)
    international['Date'] = pd.to_datetime(international['date']).dt.strftime('%Y-%m-%d') if 'date' in international.columns else international['Date']
    
    rolling_goals = get_rolling_goals(international, window=window)
    
    # Get most recent stats for this team
    team_data = rolling_goals[rolling_goals['team'] == team_name].tail(1)
    
    if len(team_data) > 0:
        return {
            'rolling_gf': team_data.iloc[0]['rolling_gf'],
            'rolling_ga': team_data.iloc[0]['rolling_ga']
        }
    else:
        return {
            'rolling_gf': 1.5,  # Default
            'rolling_ga': 1.5
        }


if __name__ == "__main__":
    # Example usage: Update rolling features for score prediction
    print("=== Automatic Rolling Feature Update for Score Prediction ===\n")
    
    # Prepare score prediction data with updated rolling features
    score_data = prepare_score_prediction_data_with_updated_rolling()
    
    print("\n✓ Score prediction features updated successfully!")
