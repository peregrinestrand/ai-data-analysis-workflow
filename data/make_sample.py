"""Generate data/orders.csv: 2,400 synthetic orders (seeded, so the file is reproducible).

The data is invented. It has one planted pattern worth finding: partner-channel
revenue in APAC drops sharply in Q3 2026 while every other segment grows a little.

    python data/make_sample.py
"""
import csv
import datetime as dt
import random
from pathlib import Path

PRODUCTS = {"Starter": 49.0, "Team": 199.0, "Business": 499.0, "Add-on Seats": 15.0, "Support Plan": 990.0}
REGIONS = ["EMEA", "NA", "APAC", "LATAM"]
CHANNELS = ["web", "partner", "sales"]


def main(path: Path = Path(__file__).with_name("orders.csv"), rows: int = 2400, seed: int = 7) -> Path:
    rng = random.Random(seed)
    start = dt.date(2026, 1, 1)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["order_id", "order_date", "region", "channel", "product", "units",
                         "unit_price", "discount", "returned"])
        for n in range(1, rows + 1):
            day = start + dt.timedelta(days=rng.randrange(273))  # Jan 1 to Sep 30
            region, channel = rng.choice(REGIONS), rng.choice(CHANNELS)
            product = rng.choice(list(PRODUCTS))
            units = rng.randint(1, 20)
            if region == "APAC" and channel == "partner" and day.month >= 7:
                units = max(1, units // 4)                     # the planted Q3 drop
            elif day.month >= 7:
                units += rng.randint(0, 2)                     # mild growth elsewhere
            discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.2])
            unit_price = round(PRODUCTS[product] * (1 - discount), 2)
            returned = rng.random() < 0.04
            writer.writerow([f"O-{n:05d}", day.isoformat(), region, channel, product, units,
                             unit_price, discount, str(returned).lower()])
    return path


if __name__ == "__main__":
    print(main())
