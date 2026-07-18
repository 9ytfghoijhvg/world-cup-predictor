"""
Clean the knockout portion of data/international_results.csv.

The file accumulated duplicate knockout rows (same game under both
'FIFA World Cup' and 'World Cup 2026' tags), NaN-score placeholder rows, and a
few fabricated games. These corrupt point-in-time rolling features (a game
counted twice skews a team's last-10 window).

This script removes every 'FIFA World Cup' / 'World Cup 2026' row dated on or
after the knockout start (June 28, 2026) and re-inserts the authoritative
knockout games exactly once. Group-stage rows (before June 28) and all other
matches are left untouched.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knockout_2026_results import as_dicts

INTL_PATH = 'data/international_results.csv'
KNOCKOUT_START = pd.Timestamp('2026-06-28')
WC_TAGS = ['FIFA World Cup', 'World Cup 2026']


def clean():
    intl = pd.read_csv(INTL_PATH)
    intl['date'] = pd.to_datetime(intl['date'])
    before = len(intl)

    # Drop knockout-window WC rows (dupes / NaN placeholders / fabricated games)
    drop_mask = (intl['date'] >= KNOCKOUT_START) & (intl['tournament'].isin(WC_TAGS))
    print(f"Removing {int(drop_mask.sum())} knockout-window WC rows (dupes/NaN/fake)")
    kept = intl[~drop_mask].copy()

    # Build clean knockout rows from the authoritative source
    rows = []
    for g in as_dicts():
        rows.append({
            'date': pd.Timestamp(g['date']),
            'home_team': g['home_team'],
            'away_team': g['away_team'],
            'home_score': g['home_score'],
            'away_score': g['away_score'],
            'tournament': 'World Cup 2026',
            'city': '',
            'country': g['venue_country'],
            'neutral': g['host_advantage'] == 0,
        })
    knockout_df = pd.DataFrame(rows)
    print(f"Inserting {len(knockout_df)} authoritative knockout games")

    cleaned = pd.concat([kept, knockout_df], ignore_index=True)
    cleaned = cleaned.sort_values('date').reset_index(drop=True)
    cleaned['date'] = cleaned['date'].dt.strftime('%Y-%m-%d')
    cleaned.to_csv(INTL_PATH, index=False)

    print(f"Rows: {before} -> {len(cleaned)}")
    wc = cleaned[cleaned['tournament'] == 'World Cup 2026']
    print(f"World Cup 2026 knockout rows now: {len(wc)}")


if __name__ == '__main__':
    clean()
