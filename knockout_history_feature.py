import pandas as pd
import numpy as np


def get_knockout_history(df):
    """
    Calculate average knockout tournament performance for each team.
    
    Scoring:
    - Round of 16 exit: 1 point
    - Quarterfinal exit: 2 points
    - Semifinal exit: 3 points
    - Final exit: 4 points
    - Tournament winner: 5 points
    """
    
    # Map round names to scores
    round_scores = {
        'Round of 16': 1,
        'Quarter-finals': 2,
        'Semi-finals': 3,
        'Final': 4,
        'Third-place match': 3,  # Third place is semi-final level
        'Second round': 2,  # Historical, similar to QF
        'Final stage': 2  # Historical, similar to QF
    }
    
    records = []
    
    for idx, match in df.iterrows():
        round_name = match['Round']
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Get the round score
        round_score = round_scores.get(round_name, 1)
        
        # Determine winner (home_score > away_score, or home won on penalties)
        home_won = (match['home_score'] > match['away_score']) or \
                   ((match['home_score'] == match['away_score']) and 
                    (match['home_penalty'] > match['away_penalty']))
        
        if home_won:
            # Home team won: gets 5, away team gets round_score
            records.append({'team': home_team, 'knockout_score': 5})
            records.append({'team': away_team, 'knockout_score': round_score})
        else:
            # Away team won: gets 5, home team gets round_score
            records.append({'team': home_team, 'knockout_score': round_score})
            records.append({'team': away_team, 'knockout_score': 5})
    
    # Convert to dataframe and average by team
    knockout_df = pd.DataFrame(records)
    knockout_history = knockout_df.groupby('team')['knockout_score'].mean().reset_index()
    knockout_history.columns = ['team', 'knockout_history_score']
    
    return knockout_history
