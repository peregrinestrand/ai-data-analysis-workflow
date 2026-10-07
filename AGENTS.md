# Data analysis rules for coding agents

This folder holds CSV files under `data/` and analysis scripts under `analysis/`.

## Never send rows you were not asked to show

- Do not `cat`, `head` or print a data file, and do not paste rows into the conversation.
- Learn the data from its profile: `python analyze.py data/<file>.csv --show-profile`.
- When a question needs data, write a script under `analysis/` that computes an aggregate and prints at most 50 rows.

## Loading data

- Load CSVs with `analyze.load(path)`. It treats only empty cells as missing; pandas' default would turn the region code `NA` (North America) into NaN.
- Columns ending in `date`, `time` or `_at` are parsed as datetimes by `load`.
- Revenue is `units * unit_price` for rows where `returned` is false.

## Answering

- Every number in an answer comes from a script you ran, with the script's path next to it.
- Say which rows were excluded (returns, missing values) and why.
- If the data cannot answer the question, say what column or period is missing.
