from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from risklab.validation import returns_matrix, allocation, confidence_horizon


@dataclass(frozen=True)
class RiskEstimate:
    method: str
    confidence: float
    horizon_days: int
    var: float
    expected_shortfall: float
    observations: int


class RiskEngine(Protocol):
    def estimate(self, returns, weights, confidence=0.99, horizon=1) -> RiskEstimate: ...


def empirical_tail(losses, confidence):
    """Generalized-inverse empirical VaR and exact fractional tail-mass ES."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all() or not 0.5 < confidence < 1:
        raise ValueError("Finite nonempty losses and valid confidence required.")
    var = float(np.quantile(losses, confidence, method="inverted_cdf"))
    ordered = np.sort(losses)[::-1]
    mass = len(losses) * (1 - confidence)
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail = float(ordered[:whole].sum())
    if fraction > 0:
        tail += fraction * ordered[whole]
    return var, tail / mass


class HistoricalEngine:
    def estimate(self, returns, weights, confidence=0.99, horizon=1):
        confidence_horizon(confidence, horizon)
        matrix = returns_matrix(returns)
        w = allocation(weights, matrix.shape[1])
        portfolio = matrix @ w
        if horizon > len(portfolio):
            raise ValueError("Historical horizon exceeds observed history.")
        losses = -np.convolve(portfolio, np.ones(horizon), "valid")
        var, es = empirical_tail(losses, confidence)
        return RiskEstimate("historical", confidence, horizon, var, es, len(losses))


class ParametricEngine:
    def estimate(self, returns, weights, confidence=0.99, horizon=1):
        confidence_horizon(confidence, horizon)
        matrix = returns_matrix(returns)
        portfolio = matrix @ allocation(weights, matrix.shape[1])
        mean = float(portfolio.mean()) * horizon
        sigma = float(portfolio.std(ddof=1)) * np.sqrt(horizon)
        z = norm.ppf(confidence)
        return RiskEstimate(
            "gaussian_parametric",
            confidence,
            horizon,
            float(-mean + z * sigma),
            float(-mean + sigma * norm.pdf(z) / (1 - confidence)),
            len(matrix),
        )


@dataclass(frozen=True)
class MonteCarloEngine:
    draws: int = 20000
    seed: int = 42
    distribution: str = "normal"
    degrees_of_freedom: float = 5.0

    def __post_init__(self):
        if (
            isinstance(self.draws, bool)
            or not isinstance(self.draws, int)
            or not np.isfinite(self.degrees_of_freedom)
            or self.draws < 1000
            or self.draws > 200000
            or self.distribution not in ["normal", "student_t"]
            or self.degrees_of_freedom <= 2
        ):
            raise ValueError("Invalid Monte Carlo configuration.")

    def estimate(self, returns, weights, confidence=0.99, horizon=1):
        confidence_horizon(confidence, horizon)
        matrix = returns_matrix(returns)
        w = allocation(weights, matrix.shape[1])
        covariance = LedoitWolf().fit(matrix).covariance_
        sigma = np.sqrt(w @ covariance @ w)
        mean = float(matrix.mean(axis=0) @ w)
        rng = np.random.default_rng(self.seed)
        daily = rng.normal(size=(self.draws, horizon))
        if self.distribution == "student_t":
            daily *= np.sqrt(
                (self.degrees_of_freedom - 2) / rng.chisquare(self.degrees_of_freedom, size=daily.shape)
            )
        losses = -(mean * horizon + sigma * daily.sum(axis=1))
        var, es = empirical_tail(losses, confidence)
        return RiskEstimate(f"monte_carlo_{self.distribution}", confidence, horizon, var, es, len(matrix))
