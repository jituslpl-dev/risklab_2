import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from risklab.validation import returns_matrix, allocation


def portfolio_statistics(returns, weights, annualization=252, decay=0.94):
    matrix = returns_matrix(returns)
    w = allocation(weights, matrix.shape[1])
    portfolio = matrix @ w
    if not 0 < decay < 1:
        raise ValueError("EWMA decay must be in (0,1).")
    wealth = np.cumprod(1 + portfolio)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / peak - 1
    variance = float(portfolio[: min(20, len(portfolio))].var(ddof=1))
    for r in portfolio:
        variance = decay * variance + (1 - decay) * r * r
    return {
        "annualized_volatility": float(portfolio.std(ddof=1) * np.sqrt(annualization)),
        "ewma_annualized_volatility": float(np.sqrt(variance * annualization)),
        "annualized_geometric_return": float(wealth[-1] ** (annualization / len(portfolio)) - 1),
        "maximum_drawdown": float(drawdown.min()),
        "terminal_wealth": float(wealth[-1]),
    }, drawdown


def pca_attribution(returns, weights, names=None):
    matrix = returns_matrix(returns)
    n = matrix.shape[1]
    w = allocation(weights, n)
    names = names or [f"asset_{i}" for i in range(n)]
    if len(names) != n or len(matrix) <= n:
        raise ValueError("Valid asset names and more observations than assets required.")
    pca = PCA(n_components=n).fit(matrix)
    eigenvalues = pca.explained_variance_
    vectors = pca.components_
    exposures = vectors @ w
    component_variances = eigenvalues * exposures**2
    covariance = vectors.T @ np.diag(eigenvalues) @ vectors
    portfolio_variance = float(w @ covariance @ w)
    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive for attribution.")
    asset_components = (vectors * w) * ((eigenvalues * exposures)[:, None])
    return {
        "portfolio_daily_variance": portfolio_variance,
        "component_variances": component_variances.tolist(),
        "component_portfolio_variance_shares": (component_variances / portfolio_variance).tolist(),
        "pca_explained_variance_ratios": pca.explained_variance_ratio_.tolist(),
        "loadings": vectors.tolist(),
        "asset_variance_contributions": dict(zip(names, map(float, asset_components.sum(axis=0)))),
        "asset_daily_volatility_contributions": dict(
            zip(names, map(float, asset_components.sum(axis=0) / np.sqrt(portfolio_variance)))
        ),
    }
