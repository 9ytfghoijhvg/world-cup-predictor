import pandas as pd


def add_group_stage_features(knockout_df, group_stage_df):
    ko = knockout_df.copy()
    
    # Merge home team group stage stats
    ko = ko.merge(group_stage_df[['team', 'points', 'goal_difference']], 
                  left_on=['home_team'], 
                  right_on=['team'], 
                  how='left')
    ko.rename(columns={'points': 'home_group_points', 'goal_difference': 'home_group_gd'}, inplace=True)
    ko = ko.drop(columns=['team'])
    
    # Merge away team group stage stats
    ko = ko.merge(group_stage_df[['team', 'points', 'goal_difference']], 
                  left_on=['away_team'], 
                  right_on=['team'], 
                  how='left')
    ko.rename(columns={'points': 'away_group_points', 'goal_difference': 'away_group_gd'}, inplace=True)
    ko = ko.drop(columns=['team'])
    
    return ko
