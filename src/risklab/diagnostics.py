"""Forecast quality and dependence-aware uncertainty, with no fitted test parameters."""

import numpy as np
from risklab.backtesting import coverage_tests


def forecast_diagnostics(forecasts, confidence=0.99, repetitions=300, block_length=5, seed=42):
    loss = forecasts.realized_loss.to_numpy()
    var = forecasts["var"].to_numpy()
    es = forecasts.expected_shortfall.to_numpy()
    if len(loss) < 20 or not np.isfinite([loss, var, es]).all() or not 0.5 < confidence < 1:
        raise ValueError("Finite forecasts with at least 20 observations required.")
    if not isinstance(repetitions, int) or not 50 <= repetitions <= 2000:
        raise ValueError("Invalid bootstrap repetitions.")
    if not isinstance(block_length, int) or not 1 <= block_length <= len(loss):
        raise ValueError("Invalid block length.")
    hits = loss > var
    # Proper quantile pinball loss; lower is better at the same confidence/sample.
    pinball = (confidence - (loss < var)) * (loss - var)
    # Population identity E[VaR + (loss-VaR)+/(1-alpha) - ES] = 0 for calibrated pairs.
    # A diagnostic residual, not a named regulatory ES backtest.
    residual = var + np.maximum(loss - var, 0) / (1 - confidence) - es
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        start = rng.integers(len(loss), size=int(np.ceil(len(loss) / block_length)))
        indices = ((start[:, None] + np.arange(block_length)) % len(loss)).ravel()[: len(loss)]
        means.append(float(residual[indices].mean()))
    return {
        **coverage_tests(hits, confidence),
        "mean_quantile_loss": float(pinball.mean()),
        "mean_es_calibration_residual": float(residual.mean()),
        "es_residual_95_block_bootstrap_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_repetitions": repetitions,
        "bootstrap_block_length": block_length,
        "es_diagnostic_note": "Zero is a calibration identity; interval is exploratory, not formal ES validation.",
    }
