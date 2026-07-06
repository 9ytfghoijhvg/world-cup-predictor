import pandas as pd
import numpy as np


def get_rolling_xg(df, window=10):
    """
    Calculate rolling average xG for each team from all international matches.
    Uses actual goals as proxy for xG (highly correlated, xG data not available for most internationals).
    
    Parameters:
    - df: DataFrame with international matches (must have Date, home_team, away_team, home_score, away_score)
    - window: Number of recent matches to average (default 10)
    
    Returns:
    - DataFrame with Date, team, rolling_xg
    """
    
    df_imputed = df.copy()
    
    # Check if xG columns exist, if not use goals as proxy
    if 'home_xg' not in df_imputed.columns or 'away_xg' not in df_imputed.columns:
        # Use actual goals as proxy for xG
        df_imputed['home_xg'] = df_imputed['home_score']
        df_imputed['away_xg'] = df_imputed['away_score']
    else:
        # If xG exists, impute missing values with actual goals
        df_imputed['home_xg'] = df_imputed['home_xg'].fillna(df_imputed['home_score'])
        df_imputed['away_xg'] = df_imputed['away_xg'].fillna(df_imputed['away_score'])
    
    # Create perspective 1: Home team
    home_view = df_imputed[['Date', 'home_team', 'home_xg']].copy()
    home_view.columns = ['Date', 'team', 'xg']
    
    # Create perspective 2: Away team
    away_view = df_imputed[['Date', 'away_team', 'away_xg']].copy()
    away_view.columns = ['Date', 'team', 'xg']
    
    # Combine both perspectives
    team_matches = pd.concat([home_view, away_view], ignore_index=True)
    team_matches = team_matches.sort_values(['team', 'Date']).reset_index(drop=True)
    
    # Calculate rolling averages per team with the specified window
    team_matches['rolling_xg'] = team_matches.groupby('team')['xg'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    return team_matches[['Date', 'team', 'rolling_xg']]


def add_rolling_xg_features(knockout_df, historical_df, window=10):
    """
    Add rolling xG features to knockout matches from all international matches.
    
    For each knockout match, looks at the last N matches BEFORE that date
    to calculate rolling average xG.
    
    Parameters:
    - knockout_df: DataFrame with knockout matches (must have Date, home_team, away_team)
    - historical_df: DataFrame with all historical international matches
    - window: Number of recent matches to average (default 10)
    
    Returns:
    - knockout_df with added features:
        - home_rolling_xg_10, away_rolling_xg_10
    """
    
    ko = knockout_df.copy()
    
    # Get rolling xG for all teams
    rolling_xg = get_rolling_xg(historical_df, window=window)
    
    # Merge for home team
    ko = ko.merge(
        rolling_xg[['Date', 'team', 'rolling_xg']], 
        left_on=['Date', 'home_team'], 
        right_on=['Date', 'team'], 
        how='left'
    )
    ko.rename(columns={'rolling_xg': f'home_rolling_xg_{window}'}, inplace=True)
    ko = ko.drop(columns=['team'])
    
    # Merge for away team
    ko = ko.merge(
        rolling_xg[['Date', 'team', 'rolling_xg']], 
        left_on=['Date', 'away_team'], 
        right_on=['Date', 'team'], 
        how='left'
    )
    ko.rename(columns={'rolling_xg': f'away_rolling_xg_{window}'}, inplace=True)
    ko = ko.drop(columns=['team'])
    
    return ko


if __name__ == "__main__":
    # Test the feature
    df = pd.read_csv("../data/international_results.csv")
    df = df[df['home_score'].notna()].copy()
    df['Date'] = pd.to_datetime(df['date'])
    
    rolling_xg = get_rolling_xg(df, window=10)
    
    # Show some examples for Brazil
    brazil = rolling_xg[rolling_xg['team'] == 'Brazil'].tail(20)
    print("\n=== Brazil's Last 20 Matches - Rolling xG (window=10) ===")
    print(brazil)
    
    # Show some examples for teams at different dates
    print("\n=== Sample Teams at Different Dates ===")
    sample = rolling_xg[rolling_xg['team'].isin(['Argentina', 'France', 'Germany'])].tail(15)
    print(sample)
