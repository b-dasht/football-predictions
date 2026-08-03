# Data Dictionary — Raw Match Data Columns

Column glossary for the CSVs in `data/raw/`, sourced from
[football-data.co.uk's official notes](https://www.football-data.co.uk/notes.txt).
Not every column below is present in every season — see the "missing columns"
warnings logged by `src.data_loader.load_raw_matches()` for which columns
exist in which seasons.

## Match Info

| Column | Meaning |
|---|---|
| `Div` | League division |
| `Date` | Match date |
| `Time` | Kick-off time |
| `HomeTeam` / `AwayTeam` | Team names |
| `FTHG` / `FTAG` | Full-time home/away goals |
| `FTR` | Full-time result (`H`/`D`/`A`) |
| `HTHG` / `HTAG` | Half-time home/away goals |
| `HTR` | Half-time result (`H`/`D`/`A`) |
| `Referee` | Match referee |

## Match Statistics

| Column | Meaning |
|---|---|
| `HS` / `AS` | Home/away shots |
| `HST` / `AST` | Home/away shots on target |
| `HHW` / `AHW` | Home/away hit woodwork |
| `HC` / `AC` | Home/away corners |
| `HF` / `AF` | Home/away fouls committed |
| `HFKC` / `AFKC` | Home/away free kicks conceded (fouls + offsides + other offenses; used instead of `HF`/`AF` where fouls data isn't available) |
| `HO` / `AO` | Home/away offsides |
| `HY` / `AY` | Home/away yellow cards |
| `HR` / `AR` | Home/away red cards |
| `HBP` / `ABP` | Home/away bookings points (10 = yellow, 25 = red) |

Note: English/Scottish yellow cards exclude the initial yellow when a second
converts it to a red (that one's counted as just the red); European
competitions count both.

## Betting Odds — Match Result (1X2)

Each bookmaker has three columns: `H` (home win), `D` (draw), `A` (away win).
A `C` suffix (e.g. `B365CH`) means **closing odds** instead of pre-match odds.

| Prefix | Bookmaker |
|---|---|
| `B365` | Bet365 — **the one this project uses as the baseline** (present across all 16 seasons pulled so far) |
| `1XB` | 1XBet |
| `BF` | Betfair |
| `BFD` | Betfred |
| `BFE` | Betfair Exchange |
| `BMGM` | BetMGM |
| `BV` | Betvictor |
| `BS` | Blue Square |
| `BW` | Bet&Win |
| `CL` | Coral |
| `GB` | Gamebookers |
| `IW` | Interwetten |
| `LB` | Ladbrokes |
| `PP` | Paddy Power |
| `PS` / `P` | Pinnacle |
| `SK` | Skybet |
| `SO` | Sporting Odds |
| `SB` | Sportingbet |
| `SJ` | Stan James |
| `SY` | Stanleybet |
| `VC` | VC Bet (now Betvictor) |
| `WH` | William Hill |
| `Max` | Market maximum odds across bookmakers |
| `Avg` | Market average odds across bookmakers |
| `Bb1X2` | Number of Betbrain bookmakers used for `BbMx`/`BbAv` |
| `BbMx` / `BbAv` | Betbrain maximum/average odds (older seasons) |

## Betting Odds — Over/Under 2.5 Goals

Suffix `>2.5` = over, `<2.5` = under. Same bookmaker prefixes as above apply
(e.g. `B365>2.5`, `Max<2.5`, `Avg>2.5`). `BbOU` = number of Betbrain
bookmakers used for the Betbrain over/under averages/maximums.

## Betting Odds — Asian Handicap

| Column pattern | Meaning |
|---|---|
| `AHh` | Market size of the handicap (home team), from 2019/20 onward |
| `{Bookmaker}AHH` / `{Bookmaker}AHA` | That bookmaker's Asian handicap home/away odds |
| `{Bookmaker}AH` | That bookmaker's size of handicap (home team) |
| `BbAH` | Number of Betbrain bookmakers used for Asian handicap averages/maximums |

## Notes

- **Bookmaker columns vary by season** — some bookmakers stopped operating or were dropped from the source over the 15+ year history; a missing column for an older/newer season is expected, not a data error.
- **Why Bet365 (`B365H`/`B365D`/`B365A`)** was chosen as the odds baseline for this project: confirmed present in all 16 downloaded seasons with complete (non-null) data in effectively every match — the only consistent bookmaker across the full history. `MaxH`/`AvgH` (market max/average) only exist from 2019/20 onward, but correlate with Bet365 at ~0.99 in every season they overlap — so Bet365 isn't a lower-quality stand-in for the market consensus, it's effectively equivalent, with far longer coverage. See `docs/EDA_FINDINGS.md` for the full check.
