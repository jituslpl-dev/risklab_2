"""Hierarchical risk parity baseline using shrinkage correlation and recursive bisection."""

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from risklab.validation import returns_matrix
from risklab.optimization import risk_shares


class HierarchicalRiskParity:
    def solve(self, returns, previous=None, budget=None):
        matrix = returns_matrix(returns)
        if budget is not None:
            raise ValueError("HRP does not implement arbitrary risk budgets.")
        covariance = LedoitWolf().fit(matrix).covariance_
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0):
            raise ValueError("Positive asset variance required.")
        n = matrix.shape[1]
        correlation = covariance / np.sqrt(np.outer(diagonal, diagonal))
        distance = np.sqrt(np.clip((1 - correlation) / 2, 0, 1))
        np.fill_diagonal(distance, 0)
        order = (
            leaves_list(linkage(squareform(distance, checks=False), method="single")).tolist()
            if n > 1
            else [0]
        )
        weights = np.ones(n)
        clusters = [order]

        def cluster_variance(indices):
            sub = covariance[np.ix_(indices, indices)]
            inverse = 1 / np.diag(sub)
            inverse /= inverse.sum()
            return float(inverse @ sub @ inverse)

        while clusters:
            next_level = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                midpoint = len(cluster) // 2
                left, right = cluster[:midpoint], cluster[midpoint:]
                left_var, right_var = cluster_variance(left), cluster_variance(right)
                alpha = right_var / (left_var + right_var)
                weights[left] *= alpha
                weights[right] *= 1 - alpha
                next_level.extend([left, right])
            clusters = next_level
        shares = risk_shares(weights, covariance)
        return {
            "weights": weights.tolist(),
            "risk_shares": shares.tolist(),
            "risk_budget_squared_error": float(np.sum((shares - 1 / n) ** 2)),
            "annualized_volatility": float(np.sqrt(252 * (weights @ covariance @ weights))),
            "turnover": None if previous is None else float(np.abs(weights - previous).sum()),
            "solver_success": True,
            "solver_iterations": 0,
            "covariance_method": "LedoitWolf",
            "cluster_order": order,
        }
