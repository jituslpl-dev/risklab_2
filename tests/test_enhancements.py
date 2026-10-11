import numpy as np
import pandas as pd
import pytest
from risklab.advanced import (
    FilteredHistoricalEngine,
    BlockBootstrapEngine,
    expected_shortfall_attribution,
    bootstrap_tail_intervals,
)
from risklab.tail_optimization import CVaROptimizer, reverse_stress
from risklab.engines import empirical_tail
from risklab.data import synthetic_market
from risklab.backtesting import rolling_var
from risklab.diagnostics import forecast_diagnostics
from risklab.hierarchical import HierarchicalRiskParity


def test_filtered_future_invariance_and_scaling():
    frame, _ = synthetic_market(300)
    w = [0.25] * 4
    engine = FilteredHistoricalEngine()
    original, _ = rolling_var(frame, w, engine, window=80)
    altered = frame.copy()
    altered.iloc[120:] *= 1.5
    changed, _ = rolling_var(altered, w, engine, window=80)
    np.testing.assert_allclose(original.iloc[:40]["var"], changed.iloc[:40]["var"])
    first = engine.estimate(frame, w)
    scaled = engine.estimate(frame * 2, w)
    assert scaled.var == pytest.approx(first.var * 2)
    assert first.expected_shortfall >= first.var
    with pytest.raises(ValueError):
        engine.estimate(frame, w, horizon=2)


def test_joint_blocks_and_horizon():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 100)
    matrix = np.c_[x, -x]
    engine = BlockBootstrapEngine(draws=1000, block_length=5)
    scenarios = engine.scenarios(matrix, 10)
    np.testing.assert_allclose(scenarios[:, 0], -scenarios[:, 1])
    assert engine.estimate(matrix, [0.5, 0.5], horizon=10).expected_shortfall == pytest.approx(0, abs=1e-15)
    np.testing.assert_array_equal(scenarios, engine.scenarios(matrix, 10))
    with pytest.raises(ValueError):
        BlockBootstrapEngine(draws=20)


def test_fractional_es_attribution_ties():
    returns = np.tile([[-0.02, -0.01], [0.01, 0.02]], (15, 1))
    result = expected_shortfall_attribution(returns, [0.4, 0.6], 0.93)
    assert sum(result["asset_contributions"]) == pytest.approx(result["expected_shortfall"])
    assert result["asset_contributions"] == pytest.approx([0.008, 0.006])


def test_cvar_lp_objective_and_constraints():
    frame, _ = synthetic_market(300)
    optimizer = CVaROptimizer(turnover_cap=0.10)
    solution = optimizer.solve(frame, previous=np.ones(4) / 4)
    w = np.array(solution["weights"])
    assert w.sum() == pytest.approx(1)
    assert np.min(w) >= 0.02 - 1e-8 and np.max(w) <= 0.60 + 1e-8
    assert solution["turnover"] <= 0.10 + 1e-8
    assert solution["lp_objective"] == pytest.approx(empirical_tail(-frame.to_numpy() @ w, 0.95)[1], abs=1e-8)
    assert solution["historical_cvar"] <= empirical_tail(-frame.mean(axis=1).to_numpy(), 0.95)[1] + 1e-8


def test_reverse_stress_reconciliation_and_minimality():
    frame, _ = synthetic_market(300)
    w = np.ones(4) / 4
    result = reverse_stress(frame, w)
    assert -w @ np.array(result["asset_shocks"]) == pytest.approx(0.10)
    assert result["mahalanobis_distance"] > 0
    with pytest.raises(ValueError):
        reverse_stress(frame, w, -1)


def test_sampling_uncertainty_reproducible():
    frame, _ = synthetic_market(300)
    first = bootstrap_tail_intervals(frame, [0.25] * 4, repetitions=50)
    assert first == bootstrap_tail_intervals(frame, [0.25] * 4, repetitions=50)
    assert first["es_95_percentile_interval"][0] <= first["es_95_percentile_interval"][1]


def test_forecast_diagnostics_exact_identity():
    # Balanced discrete loss sample: 10% at .1, 90% at zero, alpha=.9.
    frame = pd.DataFrame(
        {"realized_loss": np.tile([0.1] + [0] * 9, 10), "var": 0.0, "expected_shortfall": 0.1}
    )
    result = forecast_diagnostics(frame, 0.9, repetitions=50)
    assert result["mean_es_calibration_residual"] == pytest.approx(0, abs=1e-15)
    assert result["mean_quantile_loss"] == pytest.approx(0.009)


def test_hrp_symmetric_assets_and_positive_allocation():
    # Orthogonal equal-variance centered assets should receive equal allocation.
    x = np.tile([[1, 1], [1, -1], [-1, 1], [-1, -1]], (20, 1)) * 0.01
    result = HierarchicalRiskParity().solve(x)
    assert result["weights"] == pytest.approx([0.5, 0.5])
    frame, _ = synthetic_market(300)
    result = HierarchicalRiskParity().solve(frame)
    assert sum(result["weights"]) == pytest.approx(1)
    assert min(result["weights"]) > 0


def test_enhanced_workflow_artifacts_and_shared_dates(tmp_path):
    import json
    from risklab.enhanced_cli import enhanced_run
    from risklab.data import ASSET_CLASSES, file_hash

    frame, _ = synthetic_market(300)
    result = enhanced_run(frame, ASSET_CLASSES, tmp_path, window=252, draws=1000)
    assert len(result["forecast_comparison"]) == 6
    streams = [
        pd.read_csv(tmp_path / name)
        for name in [
            "equal_weight_returns.csv",
            "risk_parity_returns.csv",
            "cvar_returns.csv",
            "hrp_returns.csv",
        ]
    ]
    assert all(stream.date.equals(streams[0].date) for stream in streams)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert all(file_hash(tmp_path / name) == digest for name, digest in manifest.items())
    assert "enhanced_wealth.png" in (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert (tmp_path / "figures" / "enhanced_wealth.png").exists()
