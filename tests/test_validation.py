import numpy as np
import pandas as pd
import pytest
from risklab.validation import returns_matrix, allocation, dated_frame
from risklab.engines import MonteCarloEngine
from risklab.data import load_returns
from risklab.stress import StressScenario


def test_invalid_risk_inputs_are_rejected():
    with pytest.raises(ValueError):
        returns_matrix(np.full((20, 2), np.nan))
    with pytest.raises(ValueError):
        returns_matrix(np.full((20, 2), -1.0))
    with pytest.raises(ValueError):
        allocation([0.5, -0.5], 2)
    with pytest.raises(ValueError):
        allocation([1], 2)
    with pytest.raises(ValueError):
        MonteCarloEngine(draws=4000.5)
    with pytest.raises(ValueError):
        MonteCarloEngine(degrees_of_freedom=np.nan)
    dates = pd.DatetimeIndex(["2020-01-01"] * 20)
    with pytest.raises(ValueError):
        dated_frame(pd.DataFrame({"x": np.zeros(20)}, index=dates))
    scenario = StressScenario("fixed", {"equity": -0.2})
    with pytest.raises(TypeError):
        scenario.class_shocks["equity"] = -0.5


def test_price_loading_never_fills_missing_values(tmp_path):
    frame = pd.DataFrame({"asset": np.linspace(100, 120, 30)}, index=pd.bdate_range("2020-01-01", periods=30))
    frame.iloc[5] = np.nan
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index_label="Date")
    with pytest.raises(ValueError, match="complete"):
        load_returns(path, prices=True)
