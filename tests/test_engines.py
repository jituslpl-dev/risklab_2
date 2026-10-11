import numpy as np
import pytest
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from risklab.engines import empirical_tail, HistoricalEngine, ParametricEngine, MonteCarloEngine
from risklab.analytics import portfolio_statistics, pca_attribution


def test_fractional_expected_shortfall_and_ties():
    var, es = empirical_tail([1, 2, 3, 4], 0.625)
    assert var == 3 and es == pytest.approx(11 / 3)
    assert empirical_tail([1, 2, 4, 4], 0.75) == (4, 4)
    assert empirical_tail([-4, -3, -2, -1], 0.75)[0] == -2


def test_gaussian_formula_and_monte_carlo_agreement():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, size=(5000, 3))
    weights = np.ones(3) / 3
    p = returns @ weights
    estimate = ParametricEngine().estimate(returns, weights, 0.99, 5)
    assert estimate.var == pytest.approx(-5 * p.mean() + np.sqrt(5) * p.std(ddof=1) * norm.ppf(0.99))
    cov = LedoitWolf().fit(returns).covariance_
    sigma = np.sqrt(weights @ cov @ weights)
    mc = MonteCarloEngine(draws=100000).estimate(returns, weights, 0.99)
    assert mc.var == pytest.approx(-p.mean() + norm.ppf(0.99) * sigma, rel=0.04)
    assert mc.expected_shortfall >= mc.var
    assert MonteCarloEngine(draws=1000).estimate(returns, weights) == MonteCarloEngine(draws=1000).estimate(
        returns, weights
    )


def test_parametric_historical_and_student_tail_order():
    r = np.random.default_rng(7).normal(0, 0.01, size=(2000, 2))
    w = [0.5, 0.5]
    for engine in [HistoricalEngine(), ParametricEngine(), MonteCarloEngine(distribution="student_t")]:
        low = engine.estimate(r, w, 0.95)
        high = engine.estimate(r, w, 0.99)
        assert high.var >= low.var and high.expected_shortfall >= high.var
    with pytest.raises(ValueError):
        HistoricalEngine().estimate(r[:20], w, horizon=21)
    with pytest.raises(ValueError):
        MonteCarloEngine(degrees_of_freedom=2)
    with pytest.raises(ValueError):
        ParametricEngine().estimate(r, [1, 1])


def test_drawdown_includes_initial_loss_and_pca_reconciles():
    stats, _ = portfolio_statistics(np.full((20, 1), -0.01), [1])
    assert stats["maximum_drawdown"] == pytest.approx(0.99**20 - 1)
    r = np.random.default_rng(21).normal(size=(300, 4)) * 0.01
    w = np.array([0.1, 0.2, 0.3, 0.4])
    attr = pca_attribution(r, w)
    expected = float(w @ np.cov(r, rowvar=False) @ w)
    assert attr["portfolio_daily_variance"] == pytest.approx(expected)
    assert sum(attr["component_variances"]) == pytest.approx(expected)
    assert sum(attr["asset_variance_contributions"].values()) == pytest.approx(expected)
