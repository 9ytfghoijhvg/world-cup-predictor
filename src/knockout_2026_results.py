"""
Authoritative 2026 FIFA World Cup knockout-stage results.

Single source of truth, transcribed from the official Wikipedia bracket
(https://en.wikipedia.org/wiki/2026_FIFA_World_Cup). Used to (1) clean the
knockout portion of international_results.csv and (2) drive the point-in-time
backtest that logs model predictions vs. actual results.

Orientation (home/away) follows the Wikipedia match listing. `venue_country`
is used to assign host advantage (host nations: United States, Mexico, Canada).

For draws, `pen_home`/`pen_away` give the shootout result; the team that
advances is derived in `winner()`. The third-place match and final had not been
played as of the data snapshot (July 15) — their scores are marked unverified.
"""

HOST_NATIONS = {"United States", "Mexico", "Canada"}

# round, date, home, away, home_score, away_score, pen_home, pen_away, venue_country, verified
KNOCKOUT_GAMES = [
    # ---------------- Round of 32 (June 28 – July 3) ----------------
    ("Round of 32", "2026-06-28", "South Africa", "Canada", 0, 1, None, None, "United States", True),
    ("Round of 32", "2026-06-29", "Brazil", "Japan", 2, 1, None, None, "United States", True),
    ("Round of 32", "2026-06-29", "Germany", "Paraguay", 1, 1, 3, 4, "United States", True),
    ("Round of 32", "2026-06-29", "Netherlands", "Morocco", 1, 1, 2, 3, "Mexico", True),
    ("Round of 32", "2026-06-30", "Ivory Coast", "Norway", 1, 2, None, None, "United States", True),
    ("Round of 32", "2026-06-30", "France", "Sweden", 3, 0, None, None, "United States", True),
    ("Round of 32", "2026-06-30", "Mexico", "Ecuador", 2, 0, None, None, "Mexico", True),
    ("Round of 32", "2026-07-01", "England", "DR Congo", 2, 1, None, None, "United States", True),
    ("Round of 32", "2026-07-01", "Belgium", "Senegal", 3, 2, None, None, "United States", True),
    ("Round of 32", "2026-07-01", "United States", "Bosnia and Herzegovina", 2, 0, None, None, "United States", True),
    ("Round of 32", "2026-07-02", "Spain", "Austria", 3, 0, None, None, "United States", True),
    ("Round of 32", "2026-07-02", "Portugal", "Croatia", 2, 1, None, None, "Canada", True),
    ("Round of 32", "2026-07-02", "Switzerland", "Algeria", 2, 0, None, None, "Canada", True),
    ("Round of 32", "2026-07-03", "Australia", "Egypt", 1, 1, 2, 4, "United States", True),
    ("Round of 32", "2026-07-03", "Argentina", "Cape Verde", 3, 2, None, None, "United States", True),
    ("Round of 32", "2026-07-03", "Colombia", "Ghana", 1, 0, None, None, "United States", True),
    # ---------------- Round of 16 (July 4 – 7) ----------------
    ("Round of 16", "2026-07-04", "Canada", "Morocco", 0, 3, None, None, "United States", True),
    ("Round of 16", "2026-07-04", "Paraguay", "France", 0, 1, None, None, "United States", True),
    ("Round of 16", "2026-07-05", "Brazil", "Norway", 1, 2, None, None, "United States", True),
    ("Round of 16", "2026-07-05", "Mexico", "England", 2, 3, None, None, "Mexico", True),
    ("Round of 16", "2026-07-06", "Portugal", "Spain", 0, 1, None, None, "United States", True),
    ("Round of 16", "2026-07-06", "United States", "Belgium", 1, 4, None, None, "United States", True),
    ("Round of 16", "2026-07-07", "Argentina", "Egypt", 3, 2, None, None, "United States", True),
    ("Round of 16", "2026-07-07", "Switzerland", "Colombia", 0, 0, 4, 3, "Canada", True),
    # ---------------- Quarter-finals (July 9 – 11) ----------------
    ("Quarter-finals", "2026-07-09", "France", "Morocco", 2, 0, None, None, "United States", True),
    ("Quarter-finals", "2026-07-10", "Spain", "Belgium", 2, 1, None, None, "United States", True),
    ("Quarter-finals", "2026-07-11", "Argentina", "Switzerland", 3, 1, None, None, "United States", True),
    ("Quarter-finals", "2026-07-11", "Norway", "England", 1, 2, None, None, "United States", True),
    # ---------------- Semi-finals (July 14 – 15) ----------------
    ("Semi-finals", "2026-07-14", "Spain", "France", 2, 0, None, None, "United States", True),
    ("Semi-finals", "2026-07-15", "Argentina", "England", 2, 1, None, None, "United States", True),
    # ---------------- Third place (July 18) & Final (July 19) — unverified ----------------
    ("Third place", "2026-07-18", "France", "England", 1, 2, None, None, "United States", False),
    ("Final", "2026-07-19", "Spain", "Argentina", 1, 0, None, None, "United States", False),
]


def winner(game):
    """Return the advancing team for a game tuple."""
    _, _, home, away, hs, as_, ph, pa, _, _ = game
    if hs > as_:
        return home
    if as_ > hs:
        return away
    # Draw -> penalty shootout
    if ph is not None and pa is not None:
        return home if ph > pa else away
    return None


def host_advantage(game):
    """+2 if the home team is a host nation playing in its own country, -2 if
    the away team is, else 0."""
    _, _, home, away, _, _, _, _, venue_country, _ = game
    if home in HOST_NATIONS and home == venue_country:
        return 2
    if away in HOST_NATIONS and away == venue_country:
        return -2
    return 0


def as_dicts():
    """Return the games as a list of dicts for convenient consumption."""
    keys = ["round", "date", "home_team", "away_team", "home_score", "away_score",
            "pen_home", "pen_away", "venue_country", "verified"]
    games = []
    for g in KNOCKOUT_GAMES:
        d = dict(zip(keys, g))
        d["winner"] = winner(g)
        d["host_advantage"] = host_advantage(g)
        games.append(d)
    return games


if __name__ == "__main__":
    for d in as_dicts():
        pen = f" (pens {d['pen_home']}-{d['pen_away']})" if d['pen_home'] is not None else ""
        flag = "" if d["verified"] else "  [UNVERIFIED]"
        print(f"{d['round']:15} {d['date']}  {d['home_team']} {d['home_score']}-{d['away_score']} "
              f"{d['away_team']}{pen}  -> {d['winner']}  (host_adv {d['host_advantage']}){flag}")
