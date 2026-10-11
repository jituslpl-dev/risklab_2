from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from risklab.validation import dated_frame

ASSET_CLASSES = {"Equity": "equity", "Bond": "bond", "Gold": "commodity", "FX": "fx"}


def synthetic_market(rows=1512, seed=42):
    """Heavy-tailed, correlated returns across calm, high-volatility and crisis blocks."""
    if rows < 300:
        raise ValueError("At least 300 rows required for the regime demo.")
    rng = np.random.default_rng(seed)
    correlation = np.array(
        [[1, -0.2, 0.1, 0.25], [-0.2, 1, 0.1, -0.1], [0.1, 0.1, 1, -0.15], [0.25, -0.1, -0.15, 1]]
    )
    scales = np.array([0.011, 0.004, 0.009, 0.005])
    shocks = rng.normal(size=(rows, 4)) @ np.linalg.cholesky(correlation).T
    shocks *= np.sqrt(3 / rng.chisquare(5, size=(rows, 1)))
    regimes = np.array(["calm"] * rows, dtype=object)
    regimes[rows // 3 : 2 * rows // 3] = "high_volatility"
    regimes[2 * rows // 3 :] = "crisis"
    multipliers = np.where(regimes == "calm", 0.65, np.where(regimes == "high_volatility", 1.2, 2.0))
    returns = shocks * scales * multipliers[:, None] + np.array([0.00025, 0.00008, 0.0001, 0.00001])
    returns[regimes == "crisis", 0] -= 0.0006
    frame = pd.DataFrame(
        returns, index=pd.bdate_range("2018-01-01", periods=rows), columns=list(ASSET_CLASSES)
    )
    return dated_frame(frame), pd.Series(regimes, index=frame.index, name="generator_regime")


def load_returns(path, prices=False):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Input dates must be unique and sorted.")
    if prices:
        if (frame <= 0).any().any() or frame.isna().any().any():
            raise ValueError("Price levels must be positive and complete; no automatic forward-fill.")
        frame = frame.pct_change(fill_method=None).iloc[1:]
    return dated_frame(frame)


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
