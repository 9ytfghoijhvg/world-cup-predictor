import pandas as pd
import numpy as np

# Load the prepared data
df = pd.read_csv('data/knockout_matches_prepared.csv')
print(f"Loaded {len(df)} rows")

condition = (df['home_score'] == df['away_score']) & df['home_penalty'].notna()
penalty_matches = df[condition]

new = penalty_matches.assign(won = penalty_matches['home_penalty'] > penalty_matches['away_penalty'])
result = new[['home_team', 'won']]
result.rename(columns={'home_team': 'team'}, inplace=True)


new_away = penalty_matches.assign(won = penalty_matches['home_penalty'] < penalty_matches['away_penalty'])
result_away = new_away[['away_team', 'won']]
result_away.rename(columns={'away_team': 'team'}, inplace=True)

penalty_results = pd.concat([result, result_away], ignore_index=True)

matches = penalty_results.groupby('team')['won'].count()
wins = penalty_results.groupby('team')['won'].sum()
win_rate = wins/matches

penalty = {
    'team': matches.index,
    'total_matches': matches.values,
    'total_wins': wins.values,
    'win_rate': win_rate.values
}
penalty_stats = pd.DataFrame(penalty)