# Measured historical equity benchmark

Fixed 99% confidence, 252 prior observations, 6,000 simulation draws, seed 42. All six engines forecast the same 2,053 dates from the old public equity sample. Expected exceptions: 20.53. Loss is negative equal-weight portfolio return. Methods were not tuned using these results.

| Forecast | Exceptions | Exception rate | Kupiec p | Mean quantile loss |
|---|---:|---:|---:|---:|
| Historical | 25 | 1.22% | 0.338 | 0.00051168 |
| Gaussian parametric | 45 | 2.19% | 0.00000275 | 0.00051421 |
| Gaussian Monte Carlo | 48 | 2.34% | 0.000000207 | 0.00052144 |
| Filtered historical | 30 | 1.46% | 0.0494 | 0.00044006 |
| Joint block bootstrap | 27 | 1.32% | 0.171 | 0.00051247 |
| Student-t Monte Carlo | 21 | 1.02% | 0.917 | 0.00049629 |

Filtered historical has lower quantile loss on this sample but coverage is marginally rejected at 5%. Student-t coverage is not rejected; that does not prove correctness or establish future superiority. One-day bootstrap resamples the same empirical distribution, so its changes from historical estimates reflect simulation sampling rather than richer one-day dynamics.

Filtered historical ES residual mean is 0.000324, with exploratory block-bootstrap interval [-0.003847, 0.005993]. The Gaussian parametric residual is 0.014691, interval [0.006428, 0.024281]. These are calibration diagnostics, not formal regulatory ES tests. The latest-window empirical ES is 0.040707; asset contributions reconcile with error zero in the saved report.

| Walk-forward allocation | Net annual geometric return | Annual volatility | Maximum drawdown | Total L1 turnover |
|---|---:|---:|---:|---:|
| Equal weight | 16.30% | 23.67% | -49.08% | See baseline return stream |
| Constrained risk parity | 15.42% | 23.14% | -48.33% | 4.94 |
| Constrained minimum-CVaR | 11.10% | 22.54% | -43.88% | 14.59 |
| Hierarchical risk parity | 14.46% | 22.65% | -47.61% | 5.76 |

All strategies use 98 rebalances, 21-observation timing, prior 252-observation histories, holdings drift and 5 bps per traded notional. CVaR reduces drawdown in this sample but has lower return and substantially greater turnover. HRP and equal weight do not share constrained risk parity/CVaR position and turnover limits, so feasible sets differ. No outperformance claim is made.

The sample uses four surviving equities over 2007–2016, with unverified adjustment/dividend provenance. It does not establish current performance, total-return accuracy or real cross-asset generalization. Full data/source hashes, parameters, statistics, allocations and forecasts are retained under `reports/public_equities/`.
