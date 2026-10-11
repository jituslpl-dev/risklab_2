import numpy as np
import pandas as pd
import pytest
from risklab.engines import HistoricalEngine, ParametricEngine
from risklab.backtesting import rolling_var, coverage_tests
from risklab.regimes import walk_forward_risk_parity
from risklab.optimization import EqualWeightAllocator


def sample(rows=150):
    return pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (rows, 3)),
        index=pd.bdate_range("2020-01-01", periods=rows),
        columns=["a", "b", "c"],
    )


def test_forecasts_cannot_see_current_or_future_returns():
    original = sample()
    changed = original.copy()
    changed.iloc[80:] *= 3
    for engine in [HistoricalEngine(), ParametricEngine()]:
        left, _ = rolling_var(original, [1 / 3] * 3, engine, window=60)
        right, _ = rolling_var(changed, [1 / 3] * 3, engine, window=60)
        pd.testing.assert_series_equal(left["var"].iloc[:21], right["var"].iloc[:21])


def test_no_exception_sequence_is_not_independence_evidence():
    result = coverage_tests([0] * 250, 0.99)
    assert np.isfinite(result["kupiec_lr"])
    assert result["christoffersen_independence_pvalue"] is None
    assert 0 <= result["exact_binomial_pvalue"] <= 1
    with pytest.raises(ValueError):
        coverage_tests([0, 2], 0.99)
    with pytest.raises(ValueError):
        coverage_tests([0, 0.5], 0.99)


def test_clustered_exceptions_fail_independence():
    hits = np.array(([0] * 90 + [1] * 10) * 10)
    assert coverage_tests(hits, 0.9)["christoffersen_independence_pvalue"] < 0.01


def test_walk_forward_costs_and_future_independence():
    original = sample()
    changed = original.copy()
    changed.iloc[100:] *= 4
    left, records = walk_forward_risk_parity(original, EqualWeightAllocator(), window=60)
    _, new_records = walk_forward_risk_parity(changed, EqualWeightAllocator(), window=60)
    for a, b in zip(records[:2], new_records[:2]):
        assert a["date"] == b["date"] and a["regime"] == b["regime"]
        for key in ["weights", "risk_shares", "annualized_volatility", "turnover", "proxy_volatility"]:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
    assert records[0]["turnover"] == pytest.approx(1)
    assert left.transaction_cost.iloc[0] == pytest.approx(0.0005)
    assert (left.net_return <= left.gross_return + 1e-12).all()
