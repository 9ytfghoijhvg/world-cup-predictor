import pandas as pd
import numpy as np


def get_head_to_head_stats(knockout_df, historical_df):
    """
    Calculate head-to-head statistics between specific opponents.
    Looks at matches 1-2 years before each knockout match.
    
    Parameters:
    - knockout_df: Knockout matches (must have Date, home_team, away_team)
    - historical_df: All international matches
    
    Returns:
    - knockout_df with added features:
        - home_h2h_win_rate: Win rate for home team vs this opponent in last 1-2 years
        - away_h2h_win_rate: Win rate for away team vs this opponent in last 1-2 years
        - home_h2h_goal_diff: Average goal difference for home team vs opponent
        - away_h2h_goal_diff: Average goal difference for away team vs opponent
        - h2h_meeting_count: Number of meetings in lookback window
    """
    
    ko = knockout_df.copy()
    
    # Initialize new columns with defaults (neutral values)
    ko['home_h2h_win_rate'] = 0.5
    ko['away_h2h_win_rate'] = 0.5
    ko['home_h2h_goal_diff'] = 0.0
    ko['away_h2h_goal_diff'] = 0.0
    ko['h2h_meeting_count'] = 0
    
    # Ensure Date is datetime
    ko['Date'] = pd.to_datetime(ko['Date'])
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    
    print(f"\nCalculating head-to-head statistics for {len(ko)} knockout matches...")
    
    # For each knockout match
    for idx, row in ko.iterrows():
        if idx % 50 == 0:
            print(f"  Processing match {idx}/{len(ko)}...")
        
        match_date = row['Date']
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Define lookback window (1-2 years before match)
        lookback_start = match_date - pd.DateOffset(years=2)
        lookback_end = match_date - pd.DateOffset(days=1)  # Exclude match day itself
        
        # Find all matches between these two specific teams in the window
        h2h_matches = historical_df[
            (historical_df['Date'] >= lookback_start) &
            (historical_df['Date'] <= lookback_end) &
            (
                ((historical_df['home_team'] == home_team) & (historical_df['away_team'] == away_team)) |
                ((historical_df['home_team'] == away_team) & (historical_df['away_team'] == home_team))
            )
        ].copy()
        
        if len(h2h_matches) > 0:
            ko.at[idx, 'h2h_meeting_count'] = len(h2h_matches)
            
            # Calculate home team stats (team perspective, not venue)
            home_wins = 0
            home_goal_diff_sum = 0
            
            for _, match in h2h_matches.iterrows():
                if match['home_team'] == home_team:
                    # Current home team was playing at home in historical match
                    goal_diff = match['home_score'] - match['away_score']
                    if goal_diff > 0:
                        home_wins += 1
                    elif goal_diff == 0:
                        # Check for penalties if it was a knockout match
                        if 'home_penalty' in match and pd.notna(match['home_penalty']):
                            if match['home_penalty'] > match['away_penalty']:
                                home_wins += 1
                else:
                    # Current home team was playing away in historical match
                    goal_diff = match['away_score'] - match['home_score']
                    if goal_diff > 0:
                        home_wins += 1
                    elif goal_diff == 0:
                        # Check for penalties if it was a knockout match
                        if 'away_penalty' in match and pd.notna(match['away_penalty']):
                            if match['away_penalty'] > match['home_penalty']:
                                home_wins += 1
                
                home_goal_diff_sum += goal_diff
            
            # Calculate win rates and goal differences
            ko.at[idx, 'home_h2h_win_rate'] = home_wins / len(h2h_matches)
            ko.at[idx, 'away_h2h_win_rate'] = (len(h2h_matches) - home_wins) / len(h2h_matches)
            ko.at[idx, 'home_h2h_goal_diff'] = home_goal_diff_sum / len(h2h_matches)
            ko.at[idx, 'away_h2h_goal_diff'] = -home_goal_diff_sum / len(h2h_matches)
    
    print(f"✅ Head-to-head calculation complete")
    print(f"   Matches with H2H data: {(ko['h2h_meeting_count'] > 0).sum()}/{len(ko)}")
    
    return ko


if __name__ == "__main__":
    # Test the feature
    print("Loading data...")
    df_wc = pd.read_csv("../data/matches_1930_2022.csv")
    df_all = pd.read_csv("../data/international_results.csv")
    
    # Filter knockout matches
    knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                       'Quarter-finals', 'Round of 16', 'Second round', 'Final stage']
    ko = df_wc[df_wc['Round'].isin(knockout_rounds)].copy()
    
    # Filter out future matches with NA scores from international data
    df_all = df_all[df_all['home_score'].notna()].copy()
    df_all['Date'] = pd.to_datetime(df_all['date'])
    
    # Rename columns to match
    df_all = df_all.rename(columns={'date': 'Date'})
    ko['Date'] = pd.to_datetime(ko['Date'])
    
    print(f"\nKnockout matches: {len(ko)}")
    print(f"International matches: {len(df_all)}")
    
    # Calculate head-to-head stats
    ko_with_h2h = get_head_to_head_stats(ko, df_all)
    
    # Show some examples
    print("\n=== Sample Head-to-Head Statistics ===")
    sample = ko_with_h2h[['Date', 'home_team', 'away_team', 'h2h_meeting_count', 
                           'home_h2h_win_rate', 'home_h2h_goal_diff']].tail(10)
    print(sample)
    
    # Show matches with substantial H2H history
    print("\n=== Matches with Most Recent H2H History ===")
    top_h2h = ko_with_h2h[ko_with_h2h['h2h_meeting_count'] > 0].nlargest(5, 'h2h_meeting_count')
    print(top_h2h[['Date', 'home_team', 'away_team', 'h2h_meeting_count', 
                   'home_h2h_win_rate', 'home_h2h_goal_diff']])
