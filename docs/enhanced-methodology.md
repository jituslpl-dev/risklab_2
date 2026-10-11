# Enhanced methodology and architectural decisions

## Conditional historical risk

Variance initializes from the first 20 portfolio returns and updates v(t+1)=lambda*v(t)+(1-lambda)*r(t)^2. Each subsequent residual uses variance available before its observation: r(t)/sqrt(v(t)). At forecast time residuals are rescaled by final known volatility. Initial observations are excluded from residual estimation; at least 40 rows are required. Mean zero is assumed and no GARCH parameters are fitted.

## Joint block bootstrap

Seeded starting indices select consecutive asset-return rows, wrapping at the sample boundary. Assets are sampled together, preserving contemporaneous dependence. Multi-day scenarios concatenate blocks and sum returns. Default block length is five. One-day estimation resamples empirical rows and adds no temporal dynamics. Sampling intervals resample an entire history repeatedly and recompute empirical VaR/ES; percentile bounds rely on local dependence/stationarity assumptions.

## Fractional tail attribution

For tail mass m=N(1-alpha), the worst floor(m) portfolio-loss observations receive 1/m and the boundary receives remaining fractional mass/m. Apply those same weights to each asset's weighted loss. Contributions sum exactly to portfolio ES; tied portfolio-loss attribution can be nonunique, with stable observation order used here.

## CVaR linear program

Minimize z+sum(u)/(N(1-alpha)) subject to u_i >= -r_i dot w - z, u_i >= 0, sum(w)=1 and weight bounds. z is free. Optional turnover variables satisfy t_j >= plus/minus(w_j-previous_j), sum(t)<=cap. Sparse constraints avoid a dense N-by-N excess identity. HiGHS solves the program, with post-solve feasibility verification. The LP objective reconciles to empirical fractional ES. See [SciPy linprog](https://docs.scipy.org/doc/scipy/reference/optimize.linprog.html) and [Rockafellar/Uryasev publications](https://uryasev.github.io/publications/).

## HRP

Ledoit–Wolf covariance yields correlation and distance sqrt((1-rho)/2). Single linkage gives leaf ordering; recursive bisection allocates inversely to each cluster's inverse-variance portfolio variance. This is a heuristic allocator, not an exact risk-budget optimizer. Near-tied distances can change ordering.

## Reverse stress

Minimize s' Sigma^-1 s subject to -w's=L. The analytical solution is s=-L*Sigma*w/(w'Sigma*w), with distance L/sqrt(w'Sigma*w). Shrinkage covariance is used. Shocks implying simple asset returns <= -100% are rejected. Distance is not a probability; no bounded nonlinear instrument solver is substituted.

## Forecast quality

Quantile pinball loss is (alpha-1{loss<VaR})(loss-VaR); compare only identical samples and confidence. ES residual is VaR+max(loss-VaR,0)/(1-alpha)-ES. Its expected value is zero for a calibrated pair with appropriate tail treatment. The block-bootstrap interval around the sample mean is exploratory, not a named regulatory ES hypothesis test. Sparse tails and multiple comparisons limit conclusions.

## Architecture

Preserve existing numerical APIs and the original composition root. Add strategies through structural contracts, keep immutable configuration, and compare all allocators in one simulator. Reports compose focused numerical modules, while input adapters and validation remain separate from formulas. Avoid deep inheritance and unnecessary external services. Retain raw-data provenance and report hashes; deterministic synthetic CI avoids dependence on data hosts. These are research reproducibility controls, not production guarantees.
