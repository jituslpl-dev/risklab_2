"""Conditional tails, dependent scenarios and reconciling tail attribution."""

from dataclasses import dataclass
import numpy as np
from risklab.engines import RiskEstimate, empirical_tail
from risklab.validation import returns_matrix, allocation, confidence_horizon


@dataclass(frozen=True)
class FilteredHistoricalEngine:
    decay: float = 0.94

    def __post_init__(self):
        if not 0 < self.decay < 1:
            raise ValueError("EWMA decay must be in (0,1).")

    def estimate(self, returns, weights, confidence=0.99, horizon=1):
        confidence_horizon(confidence, horizon)
        if horizon != 1:
            raise ValueError("Filtered historical estimation supports one day only.")
        matrix = returns_matrix(returns)
        portfolio = matrix @ allocation(weights, matrix.shape[1])
        # Initial 20 observations initialize variance and are excluded as residuals.
        variance = max(float(np.var(portfolio[:20], ddof=1)), 1e-16)
        residuals = []
        for value in portfolio[20:]:
            residuals.append(value / np.sqrt(variance))
            variance = self.decay * variance + (1 - self.decay) * value**2
        if len(residuals) < 20:
            raise ValueError("Filtered historical estimation requires 40 observations.")
        losses = -np.asarray(residuals) * np.sqrt(variance)
        var, es = empirical_tail(losses, confidence)
        return RiskEstimate("filtered_historical", confidence, 1, var, es, len(losses))


@dataclass(frozen=True)
class BlockBootstrapEngine:
    draws: int = 6000
    block_length: int = 5
    seed: int = 42

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (self.draws, self.block_length)):
            raise ValueError("Integer draw and block counts required.")
        if not 1000 <= self.draws <= 200000 or not 1 <= self.block_length <= 252:
            raise ValueError("Invalid bootstrap configuration.")

    def scenarios(self, returns, horizon=1):
        confidence_horizon(0.99, horizon)
        matrix = returns_matrix(returns)
        if self.block_length > len(matrix):
            raise ValueError("Block length exceeds history.")
        rng = np.random.default_rng(self.seed)
        blocks = int(np.ceil(horizon / self.block_length))
        starts = rng.integers(0, len(matrix), size=(self.draws, blocks))
        indices = (starts[..., None] + np.arange(self.block_length)) % len(matrix)
        indices = indices.reshape(self.draws, -1)[:, :horizon]
        # Whole asset rows preserve contemporaneous cross-asset dependence.
        return matrix[indices].sum(axis=1)

    def estimate(self, returns, weights, confidence=0.99, horizon=1):
        confidence_horizon(confidence, horizon)
        scenarios = self.scenarios(returns, horizon)
        loss = -scenarios @ allocation(weights, scenarios.shape[1])
        var, es = empirical_tail(loss, confidence)
        return RiskEstimate("circular_block_bootstrap", confidence, horizon, var, es, len(scenarios))


def expected_shortfall_attribution(returns, weights, confidence=0.99):
    confidence_horizon(confidence, 1)
    matrix = returns_matrix(returns)
    w = allocation(weights, matrix.shape[1])
    asset_losses = -matrix * w
    losses = asset_losses.sum(axis=1)
    order = np.argsort(-losses, kind="stable")
    mass = len(matrix) * (1 - confidence)
    tail_weights = np.clip(mass - np.arange(len(matrix)), 0, 1) / mass
    contributions = tail_weights @ asset_losses[order]
    return {
        "expected_shortfall": empirical_tail(losses, confidence)[1],
        "asset_contributions": contributions.tolist(),
        "reconciliation_error": float(contributions.sum() - empirical_tail(losses, confidence)[1]),
    }


def bootstrap_tail_intervals(returns, weights, confidence=0.99, repetitions=300, block_length=5, seed=42):
    """Percentile sampling intervals using whole-row circular blocks, not model confidence."""
    confidence_horizon(confidence, 1)
    matrix = returns_matrix(returns)
    w = allocation(weights, matrix.shape[1])
    if not isinstance(repetitions, int) or not 50 <= repetitions <= 2000:
        raise ValueError("Repetitions must be an integer in [50,2000].")
    if not isinstance(block_length, int) or not 1 <= block_length <= len(matrix):
        raise ValueError("Invalid block length.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(len(matrix), size=int(np.ceil(len(matrix) / block_length)))
        indices = ((starts[:, None] + np.arange(block_length)) % len(matrix)).ravel()[: len(matrix)]
        estimates.append(empirical_tail(-matrix[indices] @ w, confidence))
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "var_95_percentile_interval": bounds[:, 0].tolist(),
        "es_95_percentile_interval": bounds[:, 1].tolist(),
        "repetitions": repetitions,
        "block_length": block_length,
        "seed": seed,
    }
