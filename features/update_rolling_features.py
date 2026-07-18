"""
Automatic rolling feature update system.

This module provides functions to:
1. Add played tournament matches to international_results.csv
2. Recalculate rolling features with tournament data included
3. Update predictions with fresh rolling stats

This ensures that rolling features (goals, xG) reflect current tournament form,
not stagnant pre-tournament statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def load_prediction_log(prediction_log_path='data/prediction_log.csv'):
    """
    Load prediction log and extract played matches with actual results.
    
    Returns:
    - DataFrame with completed matches (only those with actual_winner populated)
    """
    log = pd.read_csv(prediction_log_path)
    
    # Filter to matches that have been played (have actual results)
    played = log[log['actual_winner'].notna()].copy()
    
    if len(played) == 0:
        print("No played matches found in prediction log")
        return None
    
    return played


def extract_tournament_matches(played_matches_df):
    """
    Convert prediction log entries to international_results.csv format.
    
    Parameters:
    - played_matches_df: DataFrame from prediction log with played matches
    
    Returns:
    - DataFrame formatted for appending to international_results.csv
    """
    if played_matches_df is None or len(played_matches_df) == 0:
        return None
    
    tournament_matches = pd.DataFrame({
        'date': pd.to_datetime(played_matches_df['match_date']),
        'home_team': played_matches_df['home_team'],
        'away_team': played_matches_df['away_team'],
        'home_score': played_matches_df['actual_home_score'].astype(int),
        'away_score': played_matches_df['actual_away_score'].astype(int),
        'tournament': 'World Cup 2026',
        'city': '',  # Not tracked in prediction log
        'country': '',  # Not tracked in prediction log
        'neutral': True  # Most WC matches are neutral venues
    })
    
    return tournament_matches


def update_international_results(
    international_csv_path='data/international_results.csv',
    prediction_log_path='data/prediction_log.csv',
    backup=True
):
    """
    Add newly played tournament matches to international_results.csv.
    
    Only adds matches that:
    - Have actual results in the prediction log
    - Are not already in international_results.csv
    
    Parameters:
    - international_csv_path: Path to international_results.csv
    - prediction_log_path: Path to prediction_log.csv
    - backup: Create backup of original file before updating
    
    Returns:
    - (new_matches_df, num_added): Tuple of new matches and count added
    """
    
    # Load existing international results
    international = pd.read_csv(international_csv_path)
    international['date'] = pd.to_datetime(international['date'])
    
    # Load and extract played matches
    played = load_prediction_log(prediction_log_path)
    if played is None or len(played) == 0:
        return None, 0
    
    tournament_matches = extract_tournament_matches(played)
    
    # Find matches not already in international results.
    # Key on (home_team, away_team, score) rather than date: a given tournament
    # pairing + result is unique, and this is robust to date discrepancies that
    # would otherwise cause the same match to be appended on every run.
    international_set = set(
        zip(
            international['home_team'],
            international['away_team'],
            international['home_score'],
            international['away_score'],
        )
    )

    new_matches = []
    for idx, row in tournament_matches.iterrows():
        match_key = (row['home_team'], row['away_team'], row['home_score'], row['away_score'])
        if match_key not in international_set:
            new_matches.append(row)
    
    if len(new_matches) == 0:
        print("All played matches already in international_results.csv")
        return None, 0
    
    # Create backup
    if backup:
        backup_path = international_csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        international.to_csv(backup_path, index=False)
        print(f"✓ Created backup: {backup_path}")
    
    # Append new matches
    new_matches_df = pd.DataFrame(new_matches)
    international_updated = pd.concat([international, new_matches_df], ignore_index=True)
    international_updated = international_updated.sort_values('date').reset_index(drop=True)
    
    # Save updated file
    international_updated.to_csv(international_csv_path, index=False)
    print(f"✓ Added {len(new_matches)} tournament matches to international_results.csv")
    
    return new_matches_df, len(new_matches)


def get_updated_rolling_features(
    international_csv_path='data/international_results.csv',
    historical_knockout_csv='data/knockout_matches_prepared.csv',
    window=10
):
    """
    Calculate rolling features using updated international_results.csv.
    
    This recalculates rolling goals and xG including tournament matches played.
    
    Returns:
    - (rolling_goals_df, rolling_xg_df): DataFrames with updated rolling stats
    """
    from rolling_goals_feature import get_rolling_goals
    from rolling_xg_feature import get_rolling_xg
    
    # Load updated international results
    international = pd.read_csv(international_csv_path)
    international['Date'] = pd.to_datetime(international['date'])
    
    # Calculate rolling features with updated data
    rolling_goals = get_rolling_goals(international, window=window)
    rolling_xg = get_rolling_xg(international, window=window)
    
    print(f"✓ Recalculated rolling features with {len(international)} total matches")
    
    return rolling_goals, rolling_xg


def prepare_knockout_with_updated_rolling(
    prediction_log_path='data/prediction_log.csv',
    international_csv_path='data/international_results.csv',
    output_path='data/knockout_matches_prepared_updated.csv',
    window=10,
    backup=True
):
    """
    Prepare knockout matches with updated rolling features.
    
    This is a complete pipeline that:
    1. Updates international_results.csv with played tournament matches
    2. Recalculates rolling features
    3. Prepares knockout data with fresh rolling stats
    
    Returns:
    - DataFrame with updated knockout matches
    """
    from .penalty_feature import get_penalty_stats
    from .group_stage_feature import add_group_stage_features
    from .knockout_history_feature import get_knockout_history
    from .rolling_goals_feature import add_rolling_goals_features
    from .rolling_xg_feature import add_rolling_xg_features
    
    # Step 1: Update international results with played matches
    new_matches, count_added = update_international_results(
        international_csv_path=international_csv_path,
        prediction_log_path=prediction_log_path,
        backup=backup
    )
    
    print(f"\n--- Preparing knockout matches with updated rolling features ---")
    
    # Step 2: Load historical knockout matches (for features and flipping)
    historical_ko = pd.read_csv('data/matches_1930_2022.csv')
    
    # Step 3: Load 2026 group stage
    group_stage_2026 = pd.read_csv('data/group_stage_2026.csv')
    
    # Step 4: Load updated international results  
    international = pd.read_csv(international_csv_path)
    # Ensure consistent Date column (string format like historical data)
    international = international.rename(columns={'date': 'Date'}) if 'date' in international.columns else international
    
    # Step 5: Get knockout rounds from historical data
    knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                       'Quarter-finals', 'Round of 16', 'Second round',
                       'Final stage']
    
    ko = historical_ko[historical_ko['Round'].isin(knockout_rounds)].copy()
    
    # Create winner column: 1 = home advanced, 0 = home didn't advance
    condition = (ko['home_score'] > ko['away_score']) | \
                ((ko['home_score'] == ko['away_score']) & \
                 (ko['home_penalty'] > ko['away_penalty']))
    ko['home_advanced'] = np.where(condition, 1, 0)
    
    # Flip dataset for both perspectives
    ko_flipped = ko.copy()
    ko_flipped[['home_team', 'away_team']] = ko[['away_team', 'home_team']].values
    ko_flipped[['home_score', 'away_score']] = ko[['away_score', 'home_score']].values
    ko_flipped[['home_xg', 'away_xg']] = ko[['away_xg', 'home_xg']].values
    ko_flipped[['home_penalty', 'away_penalty']] = ko[['away_penalty', 'home_penalty']].values
    ko_flipped['home_advanced'] = 1 - ko_flipped['home_advanced']
    
    ko_expanded = pd.concat([ko, ko_flipped], ignore_index=True)
    
    # Step 6: Load and merge Elo ratings
    elo = pd.read_csv('data/elo_ratings_wc2026.csv')
    
    ko_expanded = ko_expanded.merge(elo[['year', 'country', 'rating']], 
                                     left_on=['Year', 'home_team'], 
                                     right_on=['year', 'country'], 
                                     how='left')
    ko_expanded.rename(columns={'rating': 'home_elo'}, inplace=True)
    
    ko_expanded = ko_expanded.merge(elo[['year', 'country', 'rating']], 
                                     left_on=['Year', 'away_team'], 
                                     right_on=['year', 'country'], 
                                     how='left')
    ko_expanded.rename(columns={'rating': 'away_elo'}, inplace=True)
    
    ko_expanded['elo_diff'] = ko_expanded['home_elo'] - ko_expanded['away_elo']
    
    # Step 7: Add rolling features using UPDATED international data
    ko_expanded = add_rolling_goals_features(ko_expanded, international, window=window)
    ko_expanded = add_rolling_xg_features(ko_expanded, international, window=window)
    
    # Step 8: Add host advantage
    ko_expanded['host_advantage'] = np.where(
        ko_expanded['home_team'] == ko_expanded['Host'], 2,
        np.where(ko_expanded['away_team'] == ko_expanded['Host'], -2, 0)
    )
    
    # Step 9: Add penalty stats, knockout history
    penalty_stats = get_penalty_stats(ko_expanded)
    ko_expanded = ko_expanded.merge(penalty_stats[['team', 'win_rate']], 
                                     left_on=['home_team'], 
                                     right_on=['team'], 
                                     how='left')
    ko_expanded.rename(columns={'win_rate': 'home_penalty_win_rate'}, inplace=True)
    ko_expanded = ko_expanded.drop(columns=['team'], errors='ignore')
    
    ko_expanded = ko_expanded.merge(penalty_stats[['team', 'win_rate']], 
                                     left_on=['away_team'], 
                                     right_on=['team'], 
                                     how='left')
    ko_expanded.rename(columns={'win_rate': 'away_penalty_win_rate'}, inplace=True)
    ko_expanded = ko_expanded.drop(columns=['team'], errors='ignore')
    
    knockout_history = get_knockout_history(ko)
    ko_expanded = ko_expanded.merge(knockout_history, 
                                     left_on=['home_team'], 
                                     right_on=['team'], 
                                     how='left')
    ko_expanded.rename(columns={'knockout_history_score': 'home_knockout_history'}, inplace=True)
    ko_expanded = ko_expanded.drop(columns=['team'], errors='ignore')
    
    ko_expanded = ko_expanded.merge(knockout_history, 
                                     left_on=['away_team'], 
                                     right_on=['team'], 
                                     how='left')
    ko_expanded.rename(columns={'knockout_history_score': 'away_knockout_history'}, inplace=True)
    ko_expanded = ko_expanded.drop(columns=['team'], errors='ignore')
    
    # Step 10: Add group stage features
    ko_expanded = add_group_stage_features(ko_expanded, group_stage_2026)
    
    # Save updated data
    ko_expanded.to_csv(output_path, index=False)
    print(f"✓ Saved updated knockout matches: {output_path}")
    print(f"  Total rows: {len(ko_expanded)}")
    
    return ko_expanded


if __name__ == "__main__":
    # Example usage: Update rolling features with tournament data
    print("=== Automatic Rolling Feature Update ===\n")
    
    # Step 1: Update international results
    new_matches, count = update_international_results()
    
    if count > 0:
        # Step 2: Prepare knockout matches with updated rolling features
        ko_updated = prepare_knockout_with_updated_rolling()
        print("\n✓ Rolling features updated successfully!")
        print(f"  {count} new tournament matches added")
        print(f"  Rolling stats recalculated for all predictions")
    else:
        print("\nNo new matches to process")
