import numpy as np 
import pandas as pd 
from penalty_feature import get_penalty_stats
from group_stage_feature import add_group_stage_features
from knockout_history_feature import get_knockout_history
from betting_odds import get_match_betting_odds, american_to_probability

# Load original data
df = pd.read_csv("data/matches_1930_2022.csv")
group_stage = pd.read_csv("data/group_stage_2026.csv")

knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                   'Quarter-finals', 'Round of 16', 'Second round',
                   'Final stage']

ko = df[df['Round'].isin(knockout_rounds)].copy()
print(f"Total knockout matches: {len(ko)}")
print(f"Year range: {ko['Year'].min()} to {ko['Year'].max()}")

# Create winner column: 1 = home advanced, 0 = home didn't advance
condition = (ko['home_score'] > ko['away_score']) | \
            ((ko['home_score'] == ko['away_score']) & (ko['home_penalty'] > ko['away_penalty'])) 
ko['home_advanced'] = np.where(condition, 1, 0)

print(f"\nHome wins: {(ko['home_advanced'] == 1).sum()}")
print(f"Away wins: {(ko['home_advanced'] == 0).sum()}")

# ===== ADD BETTING ODDS FEATURE =====
print("\n" + "="*70)
print("ADDING BETTING ODDS FEATURE")
print("="*70)

ko['home_betting_prob'] = np.nan
ko['odds_available'] = False

# Add betting odds for each match
for idx, row in ko.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    year = row['Year']
    
    # Get odds for this match
    odds_data = get_match_betting_odds(home_team, away_team, year)
    
    if odds_data:
        ko.at[idx, 'home_betting_prob'] = odds_data['home_probability']
        ko.at[idx, 'odds_available'] = True

odds_count = ko['odds_available'].sum()
print(f"\nMatches with betting odds available: {odds_count}/{len(ko)}")
print(f"Percentage: {100 * odds_count / len(ko):.1f}%")

# Show breakdown by year
print(f"\nOdds coverage by year:")
for year in sorted(ko['Year'].unique()):
    year_data = ko[ko['Year'] == year]
    year_odds = year_data['odds_available'].sum()
    print(f"  {int(year)}: {year_odds}/{len(year_data)}")

# Calculate betting accuracy for matches where odds are available
odds_df = ko[ko['odds_available']].copy()
if len(odds_df) > 0:
    odds_df['betting_prediction'] = (odds_df['home_betting_prob'] > 0.5).astype(int)
    betting_accuracy = (odds_df['betting_prediction'] == odds_df['home_advanced']).mean()
    print(f"\nBetting market accuracy (on matches with odds): {betting_accuracy:.1%}")
else:
    print(f"\nNo matches with odds to evaluate")

# ===== PROCEED WITH FULL PIPELINE =====
print("\n" + "="*70)
print("CONTINUING FEATURE ENGINEERING PIPELINE")
print("="*70)

# Load Elo ratings
elo = pd.read_csv('data/elo_ratings_wc2026.csv')

# Merge Elo ratings
ko = ko.merge(elo[['year', 'country', 'rating']], 
              left_on=['Year', 'home_team'], 
              right_on=['year', 'country'], 
              how='left')
ko.rename(columns={'rating': 'home_elo'}, inplace=True)
ko = ko.drop(columns=['year', 'country'])

ko = ko.merge(elo[['year', 'country', 'rating']], 
              left_on=['Year', 'away_team'], 
              right_on=['year', 'country'], 
              how='left')
ko.rename(columns={'rating': 'away_elo'}, inplace=True)
ko = ko.drop(columns=['year', 'country'])

ko['elo_diff'] = ko['home_elo'] - ko['away_elo']

print(f"✓ Added Elo ratings")

# Add penalty win rates
penalty_stats = get_penalty_stats(pd.read_csv('data/matches_1930_2022.csv'))
ko = ko.merge(penalty_stats[['team', 'win_rate']], 
              left_on='home_team', 
              right_on='team', 
              how='left')
ko.rename(columns={'win_rate': 'home_penalty_win_rate'}, inplace=True)
ko = ko.drop(columns=['team'])

ko = ko.merge(penalty_stats[['team', 'win_rate']], 
              left_on='away_team', 
              right_on='team', 
              how='left')
ko.rename(columns={'win_rate': 'away_penalty_win_rate'}, inplace=True)
ko = ko.drop(columns=['team'])

print(f"✓ Added penalty win rates")

# Add rolling xG
# Note: Using pre-calculated rolling averages from knockout data if available
matches_data = pd.read_csv('data/matches_1930_2022.csv')
matches_data['rolling_xg_10'] = matches_data.groupby('home_team')['home_xg'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)

print(f"✓ Added rolling xG (simplified - using match averages)")

# Add host advantage
ko['host_advantage'] = 0
print(f"✓ Added host advantage (defaulted to 0 for historical data)")

# Add group stage features (2026 only)
ko_2026 = ko[ko['Year'] == 2026].copy()
if len(ko_2026) > 0:
    ko_2026 = add_group_stage_features(ko_2026, group_stage)
    ko.loc[ko['Year'] == 2026, 'home_group_points'] = ko_2026['home_group_points'].values
    ko.loc[ko['Year'] == 2026, 'home_group_gd'] = ko_2026['home_group_gd'].values
    ko.loc[ko['Year'] == 2026, 'away_group_points'] = ko_2026['away_group_points'].values
    ko.loc[ko['Year'] == 2026, 'away_group_gd'] = ko_2026['away_group_gd'].values
    print(f"✓ Added 2026 group stage features")
else:
    ko['home_group_points'] = 0
    ko['home_group_gd'] = 0
    ko['away_group_points'] = 0
    ko['away_group_gd'] = 0
    print(f"✓ Added group stage features (0 for pre-2026)")

# Fill missing rolling xG values with defaults
ko['home_rolling_xg_10'] = 1.5
ko['away_rolling_xg_10'] = 1.5
print(f"✓ Added rolling xG (defaulted to 1.5)")

# Final cleanup
ko = ko.dropna(subset=['elo_diff', 'home_advanced'])

print(f"\n✓ Feature engineering complete!")
print(f"Final dataset size: {len(ko)} rows")

# Save to CSV
ko.to_csv('data/knockout_matches_prepared_with_odds.csv', index=False)
print(f"\n✓ Saved to data/knockout_matches_prepared_with_odds.csv")

# Print summary statistics
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"\nFeatures in final dataset:")
print(f"  - elo_diff: {ko['elo_diff'].notna().sum()} non-null")
print(f"  - home_rolling_xg_10: {ko['home_rolling_xg_10'].notna().sum()} non-null")
print(f"  - away_rolling_xg_10: {ko['away_rolling_xg_10'].notna().sum()} non-null")
print(f"  - host_advantage: {ko['host_advantage'].notna().sum()} non-null")
print(f"  - home_penalty_win_rate: {ko['home_penalty_win_rate'].notna().sum()} non-null")
print(f"  - away_penalty_win_rate: {ko['away_penalty_win_rate'].notna().sum()} non-null")
print(f"  - home_group_points: {ko['home_group_points'].notna().sum()} non-null")
print(f"  - home_group_gd: {ko['home_group_gd'].notna().sum()} non-null")
print(f"  - home_betting_prob: {ko['home_betting_prob'].notna().sum()} non-null ⭐ NEW")
print(f"  - home_advanced (target): {ko['home_advanced'].notna().sum()} non-null")

print(f"\nTarget distribution:")
print(f"  - Home team advanced: {(ko['home_advanced'] == 1).sum()} ({100 * (ko['home_advanced'] == 1).mean():.1f}%)")
print(f"  - Away team advanced: {(ko['home_advanced'] == 0).sum()} ({100 * (ko['home_advanced'] == 0).mean():.1f}%)")

print(f"\nBetting odds feature:")
print(f"  - Matches with odds: {ko['home_betting_prob'].notna().sum()}")
print(f"  - Coverage: {100 * ko['home_betting_prob'].notna().sum() / len(ko):.1f}%")
