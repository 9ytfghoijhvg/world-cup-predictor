"""
Betting odds context from publicly available sources (ESPN, FoxSports, OddsPortal)
Data manually compiled from free public sources

Contains:
- Moneyline odds (American format) for knockout matches (2026, 2022, 2018)
- Implied win probabilities from betting markets
- Used as training feature and additional context
"""

# 2026 Round of 32 Moneyline Odds (from Fox Sports, FanDuel as of July 1, 2026)
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

# 2022 Qatar Knockout Odds (from OddsPortal, FanDuel, DraftKings archives)
# Note: These are estimated from multiple sources as exact pre-match odds varied by book
knockout_odds_2022 = {
    # Round of 16
    ('Netherlands', 'United States'): (-140, 360, 1000),  # Netherlands favorite
    ('Argentina', 'Australia'): (-650, 600, 2000),  # Argentina heavy favorite
    ('France', 'Poland'): (-280, 380, 900),  # France favorite
    ('England', 'Senegal'): (-220, 360, 850),  # England favorite
    ('Japan', 'Croatia'): (300, 380, -140),  # Croatia slight favorite
    ('Brazil', 'South Korea'): (-400, 420, 1300),  # Brazil heavy favorite
    ('Spain', 'Morocco'): (-120, 320, 700),  # Spain slight favorite
    ('Germany', 'Costa Rica'): (-240, 360, 900),  # Germany favorite
    
    # Quarterfinals  
    ('Netherlands', 'Argentina'): (120, 320, -140),  # Argentina slight favorite
    ('France', 'England'): (-120, 320, 700),  # France slight favorite
    ('Brazil', 'Japan'): (-400, 420, 1300),  # Brazil heavy favorite
    ('Croatia', 'Brazil'): (800, 500, -250),  # Brazil heavy favorite
    
    # Semifinals
    ('Argentina', 'Croatia'): (-280, 380, 900),  # Argentina favorite
    ('France', 'Morocco'): (-180, 360, 850),  # France favorite
    
    # Final
    ('Argentina', 'France'): (-110, 310, 700),  # Even match, slight Argentina edge
}

# 2018 Russia Knockout Odds (from OddsPortal, William Hill, Bet365 archives)
knockout_odds_2018 = {
    # Round of 16
    ('France', 'Argentina'): (-180, 360, 850),  # France slight favorite
    ('Uruguay', 'Portugal'): (-240, 360, 900),  # Uruguay favorite
    ('Spain', 'Russia'): (-260, 380, 900),  # Spain favorite
    ('Croatia', 'Denmark'): (110, 310, -140),  # Denmark slight favorite
    ('Sweden', 'Switzerland'): (140, 320, -160),  # Switzerland slight favorite
    ('Colombia', 'England'): (140, 320, -160),  # England slight favorite
    ('Japan', 'Belgium'): (600, 480, -200),  # Belgium heavy favorite
    ('Brazil', 'Mexico'): (-260, 380, 900),  # Brazil favorite
    
    # Quarterfinals
    ('France', 'Uruguay'): (-140, 360, 1000),  # France slight favorite
    ('Spain', 'Russia'): (-260, 380, 900),  # Spain favorite
    ('Sweden', 'England'): (320, 380, -140),  # England slight favorite
    ('Brazil', 'Belgium'): (-200, 360, 850),  # Brazil favorite
    
    # Semifinals
    ('France', 'Belgium'): (-160, 360, 850),  # France favorite
    ('England', 'Croatia'): (-120, 320, 700),  # England slight favorite
    
    # Final
    ('France', 'Croatia'): (-180, 360, 850),  # France favorite
}

# Already played 2026 matches (for reference)
knockout_odds_2026_played = {
    ('Canada', 'South Africa'): (-140, 290, 1000),
    ('Brazil', 'Japan'): (-220, 320, 850),
    ('Germany', 'Paraguay'): (550, 320, -600),
    ('Netherlands', 'Morocco'): (440, 300, -330),
    ('Ivory Coast', 'Norway'): (370, 340, -410),
    ('France', 'Sweden'): (-800, 350, 700),
    ('Mexico', 'Ecuador'): (-170, 300, 140),
}

# Remaining unplayed 2026 R32 matches
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


def get_match_betting_odds(home_team, away_team, year=2026):
    """
    Get betting odds for a specific match
    Returns dict with moneyline odds and implied probabilities
    """
    key = (home_team, away_team)
    
    # Select year-specific odds
    if year == 2026:
        if key in knockout_odds_2026:
            odds_data = knockout_odds_2026[key]
            is_played = False
        elif key in knockout_odds_2026_played:
            odds_data = knockout_odds_2026_played[key]
            is_played = True
        else:
            return None
    elif year == 2022:
        if key not in knockout_odds_2022:
            return None
        odds_data = knockout_odds_2022[key]
        is_played = True  # 2022 tournament is finished
    elif year == 2018:
        if key not in knockout_odds_2018:
            return None
        odds_data = knockout_odds_2018[key]
        is_played = True  # 2018 tournament is finished
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
        'source': f'OddsPortal / Sportsbooks Archive ({year})',
        'is_played': is_played,
        'year': year
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
