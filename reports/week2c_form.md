# Week 2c: Does form or this-season performance help?

Same data and rules as before: features use only earlier matches; stacked logistic
regression trained on 2016-17–2023-24, compared on
**2024-25**, scored once on **2025-26** (retrained including 2024-25).

Base signals: Elo rating gap, Dixon–Coles expected-goals ratio and total. Form features
are home-minus-away differences: last-5 points, goal difference and shots-on-target
difference per game; season-so-far points and goal difference per game (also weighted
by how far into the season we are); rest days since the previous league match.

## Validation (2024-25)

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Current model (55/45 blend) | 0.9807 | 0.1994 | 0.5843 | 52.7% | 1752 |
| Base (Elo + DC signals) | 0.9802 | 0.1995 | 0.5842 | 52.9% | 1752 |
| + last-5 form | 0.9795 | 0.1992 | 0.5837 | 53.0% | 1752 |
| + this season so far | 0.9811 | 0.1997 | 0.5847 | 52.7% | 1752 |
| + rest days | 0.9806 | 0.1996 | 0.5845 | 52.9% | 1752 |
| + everything | 0.9807 | 0.1995 | 0.5845 | 53.0% | 1752 |

## Test (2025-26)

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Current model (55/45 blend) | 0.9924 | 0.2019 | 0.5918 | 52.0% | 1752 |
| Base (Elo + DC signals) | 0.9926 | 0.2020 | 0.5918 | 52.2% | 1752 |
| + last-5 form | 0.9917 | 0.2015 | 0.5910 | 51.9% | 1752 |
| + this season so far | 0.9925 | 0.2020 | 0.5918 | 52.1% | 1752 |
| + rest days | 0.9926 | 0.2020 | 0.5918 | 52.2% | 1752 |
| + everything | 0.9918 | 0.2015 | 0.5910 | 52.1% | 1752 |

## Is the difference real? (paired bootstrap, log loss, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
| + everything vs Base (Elo + DC signals) | -0.0008 | [-0.0030, +0.0014] | no clear difference |
| + last-5 form vs Base (Elo + DC signals) | -0.0008 | [-0.0029, +0.0013] | no clear difference |
| + this season so far vs Base (Elo + DC signals) | -0.0000 | [-0.0008, +0.0007] | no clear difference |
| + everything vs Current model (55/45 blend) | -0.0006 | [-0.0038, +0.0027] | no clear difference |

## Early vs late season (test, log loss)

| phase          |   matches |   Base (Elo + DC signals) |   + this season so far |   + everything |
|:---------------|----------:|--------------------------:|-----------------------:|---------------:|
| first 10 games |       480 |                    0.9808 |                 0.9806 |         0.9801 |
| rest of season |      1272 |                    0.9970 |                 0.9971 |         0.9962 |

## What the model leans on (standardised coefficients, home win vs away win)

Bigger = more influence, given everything else in the model. Features overlap a lot
(a team in good form usually also has a high rating), so small coefficients mean
"adds little beyond the others", not "irrelevant to football".

|              |   weight |
|:-------------|---------:|
| xg_ratio     |   +0.564 |
| elo_gap      |   +0.519 |
| d_form5_sotd |   +0.154 |
| season_gd_w  |   +0.154 |
| season_pts_w |   -0.149 |
| d_form5_pts  |   -0.111 |
| d_season_pts |   +0.099 |
| d_season_gd  |   -0.045 |
| d_rest_days  |   +0.018 |
| xg_total     |   -0.012 |
| d_form5_gd   |   +0.008 |
