import numpy as np
import pytest
from risklab.optimization import RiskParityOptimizer, OptimizationError
from risklab.stress import load_scenarios, StressScenario
from risklab.data import ASSET_CLASSES


def test_parity_constraints_and_infeasible_cap():
    rng = np.random.default_rng(42)
    r = rng.normal(size=(1500, 4)) * np.array([0.008, 0.009, 0.011, 0.012])
    result = RiskParityOptimizer(upper_bound=0.4, annual_volatility_cap=0.15).solve(r)
    w = np.array(result["weights"])
    assert w.sum() == pytest.approx(1)
    assert np.max(w) <= 0.4 + 1e-7 and np.min(w) >= 0.02 - 1e-7
    assert result["annualized_volatility"] <= 0.15 + 1e-6
    assert np.max(np.abs(np.array(result["risk_shares"]) - 0.25)) < 0.005
    with pytest.raises(ValueError):
        RiskParityOptimizer(upper_bound=0.2).solve(r)
    with pytest.raises(OptimizationError):
        RiskParityOptimizer(annual_volatility_cap=0.0001).solve(r)


def test_turnover_cap_is_enforced():
    r = np.random.default_rng(42).normal(size=(800, 4)) * [0.02, 0.003, 0.009, 0.01]
    result = RiskParityOptimizer(turnover_cap=0.10).solve(r, previous=[0.25] * 4)
    assert result["turnover"] <= 0.10 + 1e-6
    assert result["risk_budget_squared_error"] > 0


def test_four_stresses_reconcile_to_portfolio_loss():
    scenarios = load_scenarios("config/scenarios.json")
    assert len(scenarios) == 4
    for scenario in scenarios:
        result = scenario.evaluate(ASSET_CLASSES, [0.25] * 4)
        assert sum(result["asset_return_contributions"].values()) == pytest.approx(result["portfolio_return"])
        assert result["pnl"] == pytest.approx(result["portfolio_return"] * 1e6)
    with pytest.raises(ValueError):
        StressScenario("invalid", {"equity": -1}).evaluate({"a": "equity"}, [1])
