# Methodology and numerical conventions

## Loss, confidence and horizon

Daily asset inputs are simple fractional returns. Long-only weights are nonnegative and sum to one. Portfolio linear loss is `L = -wᵀr`, expressed as a fraction of initial portfolio notional. Positive risk numbers mean losses. VaR can be negative in an all-gain window; it is not floored to zero.

Historical VaR uses the generalized inverse of the empirical CDF (`inverted_cdf`). Expected Shortfall integrates the upper tail with total mass `n × (1 − confidence)`: take complete worst observations plus a fractional boundary observation. This avoids inflating/discarding tail mass when confidence is not an empirical quantile boundary. CVaR and Expected Shortfall are used as names for this same tail-risk measure.

Gaussian parametric risk uses sample portfolio mean and standard deviation: `VaR = -h μ + sqrt(h) σ zα`, `ES = -h μ + sqrt(h) σ φ(zα)/(1 − α)`. Mean and covariance are estimated only on the supplied window. Horizon scaling assumes independent, identically distributed daily returns.

Monte Carlo projects a Ledoit–Wolf asset covariance to portfolio variance and simulates its scalar marginal. This is equivalent to a linear portfolio under multivariate Gaussian shocks, but does not produce joint asset paths for nonlinear instruments. The optional Student-t marginal is normalized to unit variance and simulates independent daily shocks, then sums them across the requested horizon. Historical multi-day risk uses overlapping sums. All multi-day engines approximate linear P&L; they do not compound nonlinear instrument prices. Rolling statistical backtests intentionally use one-day, non-overlapping outcomes.

## Volatility, wealth and attribution

Sample volatility is annualized with 252 trading observations. EWMA uses decay 0.94 and squared returns, initialized from up to the first 20 observations in the supplied descriptive history. Drawdown includes initial wealth 1.0 in the running peak, so a first-day loss is included.

PCA is fitted on centered raw return covariance, not standardized asset correlations. Portfolio component variance is `λk (vkᵀw)²`. Asset/component contributions are `wi vki λk (vkᵀw)` and sum to portfolio variance. Euler volatility contributions divide asset variance contributions by portfolio standard deviation. Some asset contributions may be negative when an asset hedges the portfolio. This is variance/volatility attribution, not a claim that non-Gaussian CVaR decomposes along PCA components.

## VaR backtesting

Forecast for date `t` uses only observations `[t-window, t)`. The exception indicator is `realized loss > VaR`, a strict comparison. Reports include Kupiec unconditional coverage, exact binomial coverage, Christoffersen transition independence, and combined conditional coverage. A state transition model cannot be estimated when one of the two origin states is absent; such independence results are null rather than reported as evidence of independence. Low expected exception counts are flagged. These are VaR tests, not a formal Expected Shortfall validation or a Basel capital certification.

## Risk parity and trading

The objective minimizes squared differences between normalized Euler variance shares and configured risk budgets (equal shares by default). SciPy SLSQP enforces sum-to-one, long-only bounds, optional annualized volatility cap, and optional L1 turnover cap. Multiple deterministic starts are tried. Only converged, independently checked feasible solutions are returned; infeasibility raises an error. Bounds can prevent exact parity, so deviation is reported explicitly.

An analytic objective gradient is supplied. Turnover constraints use auxiliary nonnegative variables `ui >= ±(wi − previous_i)` and `sum(ui) <= cap`, an exact linear epigraph of L1 turnover. This avoids a nonsmooth absolute-value constraint at unchanged weights; actual L1 turnover is still reported and checked by regression tests.

Walk-forward estimation uses the trailing window and rebalances every 21 observations. Holdings drift with asset returns between rebalances. Costs equal 5 basis points per dollar of total bought/sold notional (`L1` turnover), charged before the rebalance day's return. Initial purchase turnover is 1.0. Equal-weight and risk-parity baselines use the same timing, drift and costs.

Regime diagnostics classify the trailing 20-observation equal-weight proxy volatility using fixed annualized thresholds of 10% and 20%. They are known at rebalance time and do not determine future returns. The generator's three regimes and this observable proxy classifier are distinct; not every generated block must map to a different observed regime.

## Stress scenarios

Four configurable scenarios shock asset-class simple returns directly. Portfolio shock is the weighted sum and P&L is that return times configurable notional. Scenario names describe hypothetical shocks, not calibrated replays of a named crisis. They do not include second-order duration/convexity, derivative repricing, margin, funding or cross-currency carry.

Primary implementation references: [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), [scikit-learn LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html), and [Basel market-risk framework](https://www.bis.org/basel_framework/standard/MAR.htm). The portfolio analytics here are research implementations, not regulatory capital calculations.
