"""Fetch an old public Plotly stock-price sample; no raw data is committed."""

from pathlib import Path
import json
import urllib.request
import pandas as pd
from risklab.data import file_hash

if __name__ == "__main__":
    output = Path("data")
    output.mkdir(exist_ok=True)
    source = "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv"
    raw = output / "plotly_stockdata.csv"
    urllib.request.urlretrieve(source, raw)
    frame = pd.read_csv(raw, parse_dates=["Date"]).set_index("Date")
    frame = frame[["MSFT", "IBM", "SBUX", "AAPL"]].sort_index()
    frame.to_csv(output / "equity_prices.csv", index_label="Date")
    classes = {column: "equity" for column in frame.columns}
    (output / "equity_classes.json").write_text(json.dumps(classes, indent=2))
    (output / "source.json").write_text(
        json.dumps(
            {
                "url": source,
                "raw_sha256": file_hash(raw),
                "note": "Historical public Plotly sample; adjustment/dividend provenance not independently verified. Equity-only; no cross-asset validation.",
            },
            indent=2,
        )
    )
    print("Saved public equity sample and provenance into data/.")
