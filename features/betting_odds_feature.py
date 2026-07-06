import pandas as pd
import numpy as np
from betting_odds import knockout_odds_2026, knockout_odds_2022, knockout_odds_2018


def moneyline_to_probability(moneyline):
    """
    Convert American moneyline odds to implied probability.
    
    Formula:
    - Negative odds (favorites): implied_prob = |moneyline| / (|moneyline| + 100)
    - Positive odds (underdogs): implied_prob = 100 / (moneyline + 100)
    
    Parameters:
    - moneyline: Integer (e.g., -180, +250)
    
    Returns:
    - Implied probability as float between 0 and 1
    
    Examples:
    - -180 (favorite) → 0.643 (64.3% implied probability)
    - +250 (underdog) → 0.286 (28.6% implied probability)
    """
    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    else:
        return 100 / (moneyline + 100)


def add_betting_odds_features(knockout_df):
    """
    Add betting odds probability features to knockout matches.
    
    Converts moneyline odds to implied probabilities, which represent
    the betting market's assessment of each team's win probability.
    This captures "wisdom of crowds" as a predictive signal.
    
    Parameters:
    - knockout_df: Knockout matches with home_team, away_team, Year (optional)
    
    Returns:
    - knockout_df with added features:
        - home_odds_prob: Implied probability from home team moneyline
        - away_odds_prob: Implied probability from away team moneyline
    
    Note: When betting odds are not available, defaults to 0.5 (neutral)
    """
    
    ko = knockout_df.copy()
    ko['home_odds_prob'] = 0.5  # Default neutral
    ko['away_odds_prob'] = 0.5
    
    # Combine all odds dictionaries
    all_odds = {}
    all_odds.update(knockout_odds_2018)
    all_odds.update(knockout_odds_2022)
    all_odds.update(knockout_odds_2026)
    
    odds_found = 0
    
    # Try to find odds for each match
    for idx, row in ko.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Try this matchup order
        match_key = (home_team, away_team)
        
        if match_key in all_odds:
            odds_tuple = all_odds[match_key]
            home_moneyline, draw_moneyline, away_moneyline = odds_tuple
            
            ko.at[idx, 'home_odds_prob'] = moneyline_to_probability(home_moneyline)
            ko.at[idx, 'away_odds_prob'] = moneyline_to_probability(away_moneyline)
            odds_found += 1
    
    print(f"✅ Betting odds features added")
    print(f"   Matches with odds data: {odds_found}/{len(ko)}")
    
    return ko


if __name__ == "__main__":
    # Test the feature
    print("Testing moneyline conversion...")
    
    # Test cases
    test_odds = [-180, +250, -500, +150, -110]
    for odds in test_odds:
        prob = moneyline_to_probability(odds)
        print(f"  Moneyline {odds:+5d} → {prob:.3f} ({prob*100:.1f}%)")
    
    # Test on actual knockout matches
    print("\nLoading knockout matches...")
    df_wc = pd.read_csv("../data/matches_1930_2022.csv")
    
    knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                       'Quarter-finals', 'Round of 16', 'Second round', 'Final stage']
    ko = df_wc[df_wc['Round'].isin(knockout_rounds)].copy()
    
    print(f"Knockout matches: {len(ko)}")
    
    # Add betting odds features
    ko_with_odds = add_betting_odds_features(ko)
    
    # Show some examples with odds
    print("\n=== Sample Matches with Betting Odds ===")
    matches_with_odds = ko_with_odds[ko_with_odds['home_odds_prob'] != 0.5]
    if len(matches_with_odds) > 0:
        sample = matches_with_odds[['home_team', 'away_team', 'home_odds_prob', 'away_odds_prob']].head(10)
        print(sample)
    else:
        print("  No matches found with betting odds in historical data")
        print("  (This is expected - odds data is limited to recent tournaments)")
