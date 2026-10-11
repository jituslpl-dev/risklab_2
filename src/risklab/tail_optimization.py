"""Scenario CVaR linear programming and minimum-distance reverse stress."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.covariance import LedoitWolf
from risklab.engines import empirical_tail
from risklab.optimization import OptimizationError, risk_shares
from risklab.validation import returns_matrix, allocation, confidence_horizon


@dataclass(frozen=True)
class CVaROptimizer:
    confidence: float = 0.95
    lower_bound: float = 0.02
    upper_bound: float = 0.60
    turnover_cap: float | None = 0.8

    def solve(self, returns, previous=None, budget=None):
        confidence_horizon(self.confidence, 1)
        matrix = returns_matrix(returns)
        m, n = matrix.shape
        if (
            not 0 <= self.lower_bound <= self.upper_bound <= 1
            or n * self.lower_bound > 1
            or n * self.upper_bound < 1
        ):
            raise ValueError("Infeasible weight bounds.")
        if budget is not None:
            raise ValueError("CVaR objective does not accept risk budgets.")
        if self.turnover_cap is not None and not 0 <= self.turnover_cap <= 2:
            raise ValueError("Invalid turnover cap.")
        previous = None if previous is None else allocation(previous, n)
        auxiliary = previous is not None and self.turnover_cap is not None
        # Variables: weights, free loss quantile z, nonnegative excesses, optional turnover u.
        size = n + 1 + m + (n if auxiliary else 0)
        objective = np.zeros(size)
        objective[n] = 1
        objective[n + 1 : n + 1 + m] = 1 / (m * (1 - self.confidence))
        inequalities = hstack(
            [
                csr_matrix(-matrix),
                csr_matrix(-np.ones((m, 1))),
                -eye(m),
                csr_matrix((m, n if auxiliary else 0)),
            ],
            format="csr",
        )
        rhs = np.zeros(m)
        if auxiliary:
            prefix = csr_matrix((n, 1 + m))
            positive = hstack([eye(n), prefix, -eye(n)])
            negative = hstack([-eye(n), prefix, -eye(n)])
            cap = csr_matrix(np.r_[np.zeros(n + 1 + m), np.ones(n)][None, :])
            inequalities = vstack([inequalities, positive, negative, cap], format="csr")
            rhs = np.r_[rhs, previous, -previous, self.turnover_cap]
        equality = csr_matrix(np.r_[np.ones(n), np.zeros(size - n)][None, :])
        result = linprog(
            objective,
            A_ub=inequalities,
            b_ub=rhs,
            A_eq=equality,
            b_eq=[1],
            bounds=[(self.lower_bound, self.upper_bound)] * n
            + [(None, None)]
            + [(0, None)] * m
            + ([(0, 1)] * n if auxiliary else []),
            method="highs",
        )
        if not result.success:
            raise OptimizationError(f"CVaR optimization failed: {result.message}")
        w = result.x[:n]
        if abs(w.sum() - 1) > 1e-7 or np.max(inequalities @ result.x - rhs) > 1e-7:
            raise OptimizationError("CVaR solution failed feasibility verification.")
        covariance = LedoitWolf().fit(matrix).covariance_
        shares = risk_shares(w, covariance)
        return {
            "weights": w.tolist(),
            "risk_shares": shares.tolist(),
            "risk_budget_squared_error": float(np.sum((shares - 1 / n) ** 2)),
            "annualized_volatility": float(np.sqrt(252 * (w @ covariance @ w))),
            "turnover": None if previous is None else float(np.abs(w - previous).sum()),
            "solver_success": True,
            "solver_iterations": int(result.nit),
            "historical_cvar": empirical_tail(-matrix @ w, self.confidence)[1],
            "lp_objective": float(result.fun),
            "covariance_method": "LedoitWolf",
        }


def reverse_stress(returns, weights, loss_target=0.10):
    """Least Mahalanobis-distance linear asset shock reaching a specified loss."""
    matrix = returns_matrix(returns)
    w = allocation(weights, matrix.shape[1])
    if not np.isfinite(loss_target) or loss_target <= 0:
        raise ValueError("Positive finite loss target required.")
    covariance = LedoitWolf().fit(matrix).covariance_
    variance = float(w @ covariance @ w)
    if variance <= 0:
        raise ValueError("Positive portfolio variance required.")
    shock = -loss_target * (covariance @ w) / variance
    if np.any(shock <= -1):
        raise ValueError("Target implies infeasible simple asset returns.")
    return {
        "loss_target": loss_target,
        "asset_shocks": shock.tolist(),
        "portfolio_loss": float(-w @ shock),
        "mahalanobis_distance": float(loss_target / np.sqrt(variance)),
        "interpretation": "Covariance-distance severity, not a calibrated event probability.",
    }
