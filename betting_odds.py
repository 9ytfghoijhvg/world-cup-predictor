"""
Betting odds context from publicly available sources (ESPN, FoxSports)
Data manually compiled from free public sources as of July 1, 2026

Contains:
- Moneyline odds (American format) for knockout matches
- Implied win probabilities from betting markets
- Used as training feature and additional context
"""

# Round of 32 Moneyline Odds (from Fox Sports, FanDuel as of July 1, 2026)
# Format: (home_team, away_team): (home_odds, draw_odds, away_odds)
# NOTE: These include both unplayed and already-played matches for historical reference
knockout_odds_2026 = {
    ('England', 'DR Congo'): (-380, 440, 1300),
    ('Belgium', 'Senegal'): (115, 210, 260),
    ('United States', 'Bosnia and Herzegovina'): (-280, 400, 800),
    ('Spain', 'Austria'): (-340, 420, 1000),
    ('Croatia', 'Portugal'): (400, 260, -130),
    ('Switzerland', 'Algeria'): (105, 220, 300),
    ('Australia', 'Egypt'): (240, 185, 150),
    ('Argentina', 'Cape Verde'): (-700, 650, 1900),
    ('Ghana', 'Colombia'): (650, 290, -200),
}

# Already played matches (for historical reference)
knockout_odds_2026_played = {
    ('Canada', 'South Africa'): (-140, 290, 1000),  # Canada won 1-0
    ('Brazil', 'Japan'): (-220, 320, 850),  # Brazil won 2-1
    ('Germany', 'Paraguay'): (550, 320, -600),  # Paraguay won 4-3 on pens
    ('Netherlands', 'Morocco'): (440, 300, -330),  # Morocco won 3-2 on pens
    ('Ivory Coast', 'Norway'): (370, 340, -410),  # Norway won 2-1
    ('France', 'Sweden'): (-800, 350, 700),  # France won 3-0
    ('Mexico', 'Ecuador'): (-170, 300, 140),  # Mexico won 2-0
}

# Remaining unplayed matches (as of July 1, 2026)
remaining_r32_matches = [
    ('England', 'DR Congo'),
    ('Belgium', 'Senegal'),
    ('United States', 'Bosnia and Herzegovina'),
    ('Spain', 'Austria'),
    ('Croatia', 'Portugal'),
    ('Switzerland', 'Algeria'),
    ('Australia', 'Egypt'),
    ('Argentina', 'Cape Verde'),
    ('Ghana', 'Colombia'),
    # TODO: Add remaining matches from bracket once group winners finalized
]


def american_to_probability(american_odds):
    """
    Convert American odds to implied probability
    Positive odds: probability = 100 / (odds + 100)
    Negative odds: probability = abs(odds) / (abs(odds) + 100)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def get_match_betting_odds(home_team, away_team):
    """
    Get betting odds for a specific match
    Returns dict with moneyline odds and implied probabilities
    """
    key = (home_team, away_team)
    
    # Check unplayed matches first, then played matches
    if key in knockout_odds_2026:
        odds_data = knockout_odds_2026[key]
    elif key in knockout_odds_2026_played:
        odds_data = knockout_odds_2026_played[key]
    else:
        return None
    
    home_odds, draw_odds, away_odds = odds_data
    
    # Convert to probabilities
    home_prob = american_to_probability(home_odds)
    draw_prob = american_to_probability(draw_odds)
    away_prob = american_to_probability(away_odds)
    
    # Normalize so they sum to 1 (accounting for sportsbook margin)
    total = home_prob + draw_prob + away_prob
    home_prob /= total
    draw_prob /= total
    away_prob /= total
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_moneyline': home_odds,
        'draw_moneyline': draw_odds,
        'away_moneyline': away_odds,
        'home_probability': home_prob,
        'draw_probability': draw_prob,
        'away_probability': away_prob,
        'source': 'Fox Sports / FanDuel (as of July 1, 2026)',
        'is_played': key in knockout_odds_2026_played
    }


def display_betting_context(home_team, away_team, model_prob):
    """
    Display betting market context alongside model prediction
    """
    odds_data = get_match_betting_odds(home_team, away_team)
    
    if not odds_data:
        return None
    
    # Market's implied win probability for home team
    market_win_prob = odds_data['home_probability']
    
    output = []
    output.append("")
    output.append("-" * 70)
    output.append("Market Context (Betting Odds):")
    
    if odds_data['is_played']:
        output.append(f"⚠️  Match already played")
    else:
        # Show moneyline
        home_line = f"{odds_data['home_moneyline']:+d}" if odds_data['home_moneyline'] > 0 else f"{odds_data['home_moneyline']}"
        away_line = f"{odds_data['away_moneyline']:+d}" if odds_data['away_moneyline'] > 0 else f"{odds_data['away_moneyline']}"
        
        output.append(f"  Moneyline: {home_team} {home_line} | Draw {odds_data['draw_moneyline']:+d} | {away_team} {away_line}")
        output.append(f"  Market probability: {home_team} {market_win_prob:.1%} | {away_team} {odds_data['away_probability']:.1%}")
        
        # Compare to model
        diff = (model_prob - market_win_prob) * 100
        if abs(diff) < 5:
            output.append(f"  ✓ Model aligns with market ({diff:+.1f} percentage points)")
        elif diff > 5:
            output.append(f"  ⬆️  Model favors {home_team} more than market ({diff:+.1f} pp)")
        else:
            output.append(f"  ⬇️  Model favors {away_team} more than market ({abs(diff):+.1f} pp)")
        
        output.append(f"  Source: Fox Sports / FanDuel (as of July 1, 2026)")
    
    output.append("-" * 70)
    
    return "\n".join(output)
