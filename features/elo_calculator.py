"""
Calculate Elo ratings for all international football teams from match history.

Elo formula:
  New Rating = Old Rating + K × (Actual - Expected)
  
Where:
  - K = weight factor (varies by match importance)
  - Actual = 1 (win), 0.5 (draw), 0 (loss)
  - Expected = 1 / (1 + 10^((opponent_rating - your_rating) / 400))
"""

import pandas as pd
import numpy as np


class EloCalculator:
    def __init__(self, base_rating=1500):
        """
        Initialize Elo calculator.
        
        Parameters:
        - base_rating: Starting rating for new teams (default 1500)
        """
        self.base_rating = base_rating
        self.ratings = {}  # {team: rating}
        
    def get_rating(self, team):
        """Get current rating for a team (or base rating if new)"""
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
        return self.ratings[team]
    
    def expected_result(self, rating_a, rating_b):
        """
        Calculate expected result for team A against team B.
        Returns probability between 0 and 1.
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a, team_b, score_a, score_b, 
                       tournament='Friendly', home_advantage=0):
        """
        Update ratings after a match.
        
        Parameters:
        - team_a, team_b: Team names
        - score_a, score_b: Goals scored
        - tournament: Tournament type (determines K factor)
        - home_advantage: Rating boost for home team (typically 0-100)
        
        Returns:
        - (new_rating_a, new_rating_b, rating_change_a, rating_change_b)
        """
        # Get current ratings
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Apply home advantage
        rating_a_adjusted = rating_a + home_advantage
        
        # Calculate expected results
        expected_a = self.expected_result(rating_a_adjusted, rating_b)
        expected_b = 1 - expected_a
        
        # Determine actual result
        if score_a > score_b:
            actual_a = 1.0
            actual_b = 0.0
        elif score_a < score_b:
            actual_a = 0.0
            actual_b = 1.0
        else:  # Draw
            actual_a = 0.5
            actual_b = 0.5
        
        # Goal difference multiplier (bigger wins = bigger rating change)
        goal_diff = abs(score_a - score_b)
        if goal_diff <= 1:
            gd_multiplier = 1.0
        elif goal_diff == 2:
            gd_multiplier = 1.5
        else:  # 3+
            gd_multiplier = (11 + goal_diff) / 8
        
        # Get K factor based on tournament importance
        k = self.get_k_factor(tournament) * gd_multiplier
        
        # Update ratings
        change_a = k * (actual_a - expected_a)
        change_b = k * (actual_b - expected_b)
        
        new_rating_a = rating_a + change_a
        new_rating_b = rating_b + change_b
        
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        
        return new_rating_a, new_rating_b, change_a, change_b
    
    def get_k_factor(self, tournament):
        """
        Get K factor based on tournament importance.
        
        Official K values:
        - World Cup, Olympic Games (1908-1980): 60
        - Continental championship and intercontinental tournaments: 50
        - World Cup and Continental qualifiers and major tournaments: 40
        - All other tournaments: 30
        - Friendly matches: 20
        """
        tournament_lower = str(tournament).lower()
        
        # World Cup or Olympics (1908-1980)
        if 'world cup' in tournament_lower and 'qualif' not in tournament_lower:
            return 60
        if 'olympic' in tournament_lower:
            return 60
            
        # Continental championships and intercontinental tournaments
        # Euro, Copa America, Africa Cup, Asian Cup, Gold Cup, Confederations Cup
        if any(x in tournament_lower for x in ['euro', 'copa america', 'africa cup', 
                                                 'asian cup', 'gold cup', 'confederation']):
            return 50
            
        # World Cup qualifiers, Continental qualifiers, major tournaments
        if 'qualif' in tournament_lower:
            return 40
        if any(x in tournament_lower for x in ['nations league', 'nations cup']):
            return 40
            
        # Friendly matches
        if 'friendly' in tournament_lower:
            return 20
            
        # All other tournaments (default)
        return 30


def calculate_elo_from_matches(matches_df, start_year=None, end_year=None):
    """
    Calculate Elo ratings from a dataframe of international matches.
    
    Parameters:
    - matches_df: DataFrame with columns: date, home_team, away_team, 
                  home_score, away_score, tournament, neutral
    - start_year: Optional year to start from (filters matches)
    - end_year: Optional year to end at (filters matches)
    
    Returns:
    - DataFrame with columns: team, rating, matches_played
    """
    calc = EloCalculator(base_rating=1500)
    
    # Filter by year if specified
    df = matches_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if start_year:
        df = df[df['date'].dt.year >= start_year]
    if end_year:
        df = df[df['date'].dt.year <= end_year]
    
    # Sort by date
    df = df.sort_values('date')
    
    print(f"Processing {len(df)} matches...")
    
    # Process each match
    for idx, match in df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        tournament = match.get('tournament', 'Friendly')
        neutral = match.get('neutral', False)
        
        # Skip if scores are missing
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        
        # Home advantage (100 points if not neutral venue)
        home_adv = 0 if neutral else 100
        
        # Update ratings
        calc.update_ratings(
            home_team, away_team,
            home_score, away_score,
            tournament=tournament,
            home_advantage=home_adv
        )
    
    # Create results dataframe
    results = []
    for team, rating in calc.ratings.items():
        results.append({
            'team': team,
            'rating': int(rating),
        })
    
    results_df = pd.DataFrame(results).sort_values('rating', ascending=False)
    
    return results_df


if __name__ == "__main__":
    # Test with international results
    print("="*70)
    print("CALCULATING ELO RATINGS FROM INTERNATIONAL MATCH HISTORY")
    print("="*70)
    
    # Load data
    df = pd.read_csv('../data/international_results.csv')
    print(f"\nLoaded {len(df)} international matches")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Calculate Elo ratings up to 2022 (for training data)
    print("\n" + "="*70)
    print("CALCULATING ELO RATINGS (up to 2022)")
    print("="*70)
    
    elo_2022 = calculate_elo_from_matches(df, end_year=2022)
    
    print(f"\n✅ Calculated ratings for {len(elo_2022)} teams")
    print("\nTop 20 teams by Elo (as of end of 2022):")
    print(elo_2022.head(20).to_string(index=False))
    
    # Save to CSV
    output_path_2022 = '../data/elo_ratings_all_teams_2022.csv'
    elo_2022.to_csv(output_path_2022, index=False)
    print(f"\n✅ Saved to {output_path_2022}")
    
    # Calculate Elo ratings up to 2026 (for predictions)
    print("\n" + "="*70)
    print("CALCULATING ELO RATINGS (up to 2026)")
    print("="*70)
    
    elo_2026 = calculate_elo_from_matches(df, end_year=2026)
    
    print(f"\n✅ Calculated ratings for {len(elo_2026)} teams")
    print("\nTop 20 teams by Elo (as of June 2026):")
    print(elo_2026.head(20).to_string(index=False))
    
    # Save to CSV
    output_path_2026 = '../data/elo_ratings_all_teams_2026.csv'
    elo_2026.to_csv(output_path_2026, index=False)
    print(f"\n✅ Saved to {output_path_2026}")
    
    # Check if previously missing teams are now included
    missing_teams = ['Russia', 'Denmark', 'Poland', 'Serbia', 'Iceland', 'Peru', 'Nigeria', 'Costa Rica']
    print("\n" + "="*70)
    print("VERIFICATION - Previously Missing Teams (2022)")
    print("="*70)
    for team in missing_teams:
        team_rating = elo_2022[elo_2022['team'] == team]
        if len(team_rating) > 0:
            print(f"✅ {team:15s}: {team_rating.iloc[0]['rating']}")
        else:
            print(f"❌ {team:15s}: Not found")
