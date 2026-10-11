from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from risklab.validation import returns_matrix, allocation


class OptimizationError(RuntimeError):
    pass


class EqualWeightAllocator:
    """Comparable baseline under the same walk-forward trading conventions."""

    def solve(self, returns, previous=None, budget=None):
        matrix = returns_matrix(returns)
        w = np.ones(matrix.shape[1]) / matrix.shape[1]
        covariance = LedoitWolf().fit(matrix).covariance_
        return {
            "weights": w.tolist(),
            "risk_shares": risk_shares(w, covariance).tolist(),
            "risk_budget_squared_error": float(np.sum((risk_shares(w, covariance) - w) ** 2)),
            "annualized_volatility": float(np.sqrt(252 * (w @ covariance @ w))),
            "turnover": None if previous is None else float(np.abs(w - previous).sum()),
            "solver_success": True,
            "solver_iterations": 0,
            "covariance_method": "LedoitWolf",
        }


def risk_shares(weights, covariance):
    variance = float(weights @ covariance @ weights)
    if variance <= 0:
        raise ValueError("Positive portfolio variance required.")
    return weights * (covariance @ weights) / variance


@dataclass(frozen=True)
class RiskParityOptimizer:
    lower_bound: float = 0.02
    upper_bound: float = 0.60
    annual_volatility_cap: float | None = None
    turnover_cap: float | None = None
    annualization: int = 252

    def solve(self, returns, previous=None, budget=None):
        matrix = returns_matrix(returns)
        n = matrix.shape[1]
        if (
            not 0 <= self.lower_bound <= self.upper_bound <= 1
            or n * self.lower_bound > 1
            or n * self.upper_bound < 1
        ):
            raise ValueError("Infeasible long-only weight bounds.")
        if self.annual_volatility_cap is not None and self.annual_volatility_cap <= 0:
            raise ValueError("Volatility cap must be positive.")
        if self.turnover_cap is not None and not 0 <= self.turnover_cap <= 2:
            raise ValueError("Turnover cap must be in [0,2].")
        previous = None if previous is None else allocation(previous, n)
        target = np.ones(n) / n if budget is None else allocation(budget, n)
        covariance = LedoitWolf().fit(matrix).covariance_

        auxiliary = self.turnover_cap is not None and previous is not None
        dimensions = 2 * n if auxiliary else n

        def objective(x):
            w = x[:n]
            return float(np.sum((risk_shares(w, covariance) - target) ** 2))

        def gradient(x):
            w = x[:n]
            marginal = covariance @ w
            variance = float(w @ marginal)
            shares = w * marginal / variance
            jacobian = (
                np.diag(marginal / variance)
                + (w[:, None] * covariance) / variance
                - 2 * np.outer(shares, marginal) / variance
            )
            result = np.zeros(dimensions)
            result[:n] = 2 * jacobian.T @ (shares - target)
            return result

        constraints = [{"type": "eq", "fun": lambda x: x[:n].sum() - 1}]
        if self.annual_volatility_cap is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: self.annual_volatility_cap**2
                    - self.annualization * (x[:n] @ covariance @ x[:n]),
                }
            )
        if auxiliary:
            # Exact epigraph of L1 turnover: u >= ±(w-previous), sum(u) <= cap.
            # Linear constraints avoid the nonsmooth abs() constraint at unchanged weights.
            constraints.extend(
                [
                    {"type": "ineq", "fun": lambda x: self.turnover_cap - x[n:].sum()},
                    {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - previous)},
                    {"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - previous)},
                ]
            )
        starts = [np.ones(n) / n, 1 / np.sqrt(np.diag(covariance))]
        starts[1] /= starts[1].sum()
        if previous is not None:
            starts.append(previous)
        results = []
        for start in starts:
            initial = np.r_[start, np.abs(start - previous)] if auxiliary else start
            result = minimize(
                objective,
                initial,
                jac=gradient,
                method="SLSQP",
                bounds=[(self.lower_bound, self.upper_bound)] * n + ([(0.0, 1.0)] * n if auxiliary else []),
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if (
                result.success
                and abs(result.x[:n].sum() - 1) < 1e-7
                and all(
                    np.all(np.asarray(c["fun"](result.x)) >= -1e-7)
                    for c in constraints
                    if c["type"] == "ineq"
                )
            ):
                if (
                    np.min(result.x[:n]) >= self.lower_bound - 1e-7
                    and np.max(result.x[:n]) <= self.upper_bound + 1e-7
                ):
                    results.append(result)
        if not results:
            raise OptimizationError(
                "No feasible converged solution; relax constraints or inspect covariance."
            )
        result = min(results, key=lambda r: r.fun)
        w = result.x[:n]
        return {
            "weights": w.tolist(),
            "risk_shares": risk_shares(w, covariance).tolist(),
            "risk_budget_squared_error": float(result.fun),
            "annualized_volatility": float(np.sqrt(self.annualization * (w @ covariance @ w))),
            "turnover": None if previous is None else float(np.abs(w - previous).sum()),
            "solver_success": True,
            "solver_iterations": int(result.nit),
            "covariance_method": "LedoitWolf",
        }
