# Question: Which region and channel lost the most revenue from Q2 to Q3?
# Written by hand: the shape analyze.py asks the model for, kept as a reference answer.
def analyze(df):
    import pandas as pd

    kept = df[~df["returned"]]
    frame = pd.DataFrame({
        "region": kept["region"],
        "channel": kept["channel"],
        "quarter": kept["order_date"].dt.quarter,
        "revenue": kept["units"] * kept["unit_price"],
    })
    table = frame.pivot_table(index=["region", "channel"], columns="quarter", values="revenue", aggfunc="sum")
    table = table.rename(columns={2: "q2_revenue", 3: "q3_revenue"})[["q2_revenue", "q3_revenue"]]
    table["change_pct"] = (table["q3_revenue"] / table["q2_revenue"] - 1) * 100
    worst = table.sort_values("change_pct").head(3).round(1).reset_index()
    return {"largest_q2_to_q3_revenue_drops": worst.to_dict(orient="records")}
