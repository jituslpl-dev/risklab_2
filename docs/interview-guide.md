# Interview walkthrough

1. Explain why VaR specifies a loss quantile and Expected Shortfall measures severity beyond that quantile. State the confidence and loss-sign convention.
2. Demonstrate the fractional empirical tail test. Explain why a naive `loss >= VaR` average can mishandle quantile boundaries and ties.
3. Compare historical, Gaussian and Student-t assumptions. Explain covariance shrinkage and Monte Carlo sampling error.
4. Show a rolling forecast and the future-perturbation test. Explain why overlapping multi-day outcomes complicate standard independence tests.
5. Interpret coverage tests honestly: a non-rejection is not proof, and sparse tail events reduce power.
6. Reconcile PCA component and asset contributions to variance. Distinguish covariance PCA from correlation PCA and variance attribution from CVaR attribution.
7. Show constraints, solver diagnostics and an infeasible-cap test. Explain why bounds/turnover caps can prevent equal risk contributions.
8. Compare risk parity with equal weight under identical rebalancing and trading-cost assumptions.
9. Walk through four stress configurations. Explain what direct return shocks omit relative to full instrument repricing.
10. Discuss historical sample provenance, survivorship bias, missing dividends, market-regime shifts and why old equity samples cannot validate cross-asset performance.

Design patterns: risk engines implement a shared structural `RiskEngine` protocol; each estimator is a strategy. Immutable dataclasses hold method/policy configuration. The experiment runner composes engines, backtests, attribution and optimizers. A baseline allocator shares the optimizer's `solve` contract. Numeric validation and data loading are separate from finance calculations. Exceptions expose invalid inputs and infeasible optimizations instead of silently repairing them.
# RiskLab 2 additional interview topics

Explain why conditional filtered risk can improve quantile loss yet still fail coverage, and why a simulation technique does not fix an incorrect distribution assumption. Derive the sparse CVaR LP, the exact L1 turnover epigraph and reverse-stress closed form. Show fractional tail attribution reconciliation. Explain circular-block dependence, sampling uncertainty and why an exploratory ES residual is not formal regulatory validation. Compare HRP's clustering heuristic with risk parity's constrained objective. Discuss why CVaR reduced sample drawdown but lowered return and increased turnover. Demonstrate future-perturbation invariance tests and the shared cost/drift simulator. Describe Strategy, Adapter, composition and immutable configuration using actual classes rather than abstract pattern names.

