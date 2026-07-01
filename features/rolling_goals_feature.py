import pandas as pd
import numpy as np


def get_rolling_goals(df, window=10):
    """
    Calculate rolling average goals for/against for each team.
    Uses a sliding window approach to avoid data leakage.
    
    Parameters:
    - df: DataFrame with ALL matches (not just group stage)
    - window: Number of recent matches to average (default 10)
    
    Returns:
    - DataFrame with Date, team, rolling_gf, rolling_ga
    """
    
    # Create perspective 1: Home team
    home_view = df[['Date', 'home_team', 'home_score', 'away_score']].copy()
    home_view.columns = ['Date', 'team', 'goals_for', 'goals_against']
    
    # Create perspective 2: Away team (flip goals)
    away_view = df[['Date', 'away_team', 'away_score', 'home_score']].copy()
    away_view.columns = ['Date', 'team', 'goals_for', 'goals_against']
    
    # Combine both perspectives
    team_matches = pd.concat([home_view, away_view], ignore_index=True)
    team_matches = team_matches.sort_values(['team', 'Date']).reset_index(drop=True)
    
    # Calculate rolling averages per team with the specified window
    team_matches['rolling_gf'] = team_matches.groupby('team')['goals_for'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    team_matches['rolling_ga'] = team_matches.groupby('team')['goals_against'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    return team_matches[['Date', 'team', 'rolling_gf', 'rolling_ga']]


def add_rolling_goals_features(knockout_df, historical_df, window=10):
    """
    Add rolling goals features to knockout matches.
    
    For each knockout match, looks at the last N matches BEFORE that date
    to calculate rolling average goals for/against.
    
    Parameters:
    - knockout_df: DataFrame with knockout matches
    - historical_df: DataFrame with all historical matches
    - window: Number of recent matches to average (default 10)
    
    Returns:
    - knockout_df with added features:
        - home_rolling_gf_10, home_rolling_ga_10
        - away_rolling_gf_10, away_rolling_ga_10
    """
    
    ko = knockout_df.copy()
    
    # Get rolling goals for all teams
    rolling_goals = get_rolling_goals(historical_df, window=window)
    
    # Merge for home team
    ko = ko.merge(
        rolling_goals[['Date', 'team', 'rolling_gf', 'rolling_ga']], 
        left_on=['Date', 'home_team'], 
        right_on=['Date', 'team'], 
        how='left'
    )
    ko.rename(columns={
        'rolling_gf': f'home_rolling_gf_{window}',
        'rolling_ga': f'home_rolling_ga_{window}'
    }, inplace=True)
    ko = ko.drop(columns=['team'])
    
    # Merge for away team
    ko = ko.merge(
        rolling_goals[['Date', 'team', 'rolling_gf', 'rolling_ga']], 
        left_on=['Date', 'away_team'], 
        right_on=['Date', 'team'], 
        how='left'
    )
    ko.rename(columns={
        'rolling_gf': f'away_rolling_gf_{window}',
        'rolling_ga': f'away_rolling_ga_{window}'
    }, inplace=True)
    ko = ko.drop(columns=['team'])
    
    return ko

if __name__ == "__main__":
    # Test the feature
    df = pd.read_csv("../data/matches_1930_2022.csv")
    
    rolling_goals = get_rolling_goals(df, window=10)
    
    # Show some examples for Brazil
    brazil = rolling_goals[rolling_goals['team'] == 'Brazil'].tail(20)
    print("\n=== Brazil's Last 20 Matches - Rolling Goals (window=10) ===")
    print(brazil)
    
    # Show some examples for teams at different dates
    print("\n=== Sample Teams at Different Dates ===")
    sample = rolling_goals[rolling_goals['team'].isin(['Argentina', 'France', 'Germany'])].tail(15)
    print(sample)
