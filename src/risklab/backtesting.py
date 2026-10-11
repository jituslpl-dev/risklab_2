from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, binomtest
from risklab.validation import dated_frame, allocation


def _bernoulli_log(successes, failures, p):
    return float(xlogy(successes, p) + xlogy(failures, 1 - p))


def coverage_tests(exceptions, confidence):
    hits = np.asarray(exceptions)
    if hits.ndim != 1 or len(hits) < 2 or not np.isin(hits, [0, 1]).all() or not 0.5 < confidence < 1:
        raise ValueError("Binary exception sequence and valid confidence required.")
    hits = hits.astype(int)
    n = len(hits)
    x = int(hits.sum())
    q = 1 - confidence
    lr_uc = max(0.0, 2 * (_bernoulli_log(x, n - x, x / n) - _bernoulli_log(x, n - x, q)))
    result = {
        "observations": n,
        "exceptions": x,
        "expected_exceptions": n * q,
        "exception_rate": x / n,
        "kupiec_lr": lr_uc,
        "kupiec_pvalue": float(chi2.sf(lr_uc, 1)),
        "exact_binomial_pvalue": float(binomtest(x, n, q).pvalue),
        "christoffersen_independence_pvalue": None,
        "conditional_coverage_pvalue": None,
        "warning": "Low expected exception count: asymptotic tests may have low power."
        if n * q < 10
        else None,
    }
    counts = np.zeros((2, 2), dtype=int)
    for left, right in zip(hits[:-1], hits[1:]):
        counts[left, right] += 1
    n00, n01, n10, n11 = counts.ravel()
    if n00 + n01 > 0 and n10 + n11 > 0:
        pooled = (n01 + n11) / counts.sum()
        p0 = n01 / (n00 + n01)
        p1 = n11 / (n10 + n11)
        alternate = _bernoulli_log(n01, n00, p0) + _bernoulli_log(n11, n10, p1)
        null = _bernoulli_log(n01 + n11, n00 + n10, pooled)
        lr_ind = max(0.0, 2 * (alternate - null))
        result.update(
            christoffersen_independence_pvalue=float(chi2.sf(lr_ind, 1)),
            conditional_coverage_pvalue=float(chi2.sf(lr_uc + lr_ind, 2)),
        )
    else:
        result["independence_note"] = "Cannot estimate both transition states; no independence claim."
    return result


def rolling_var(returns, weights, engine, window=252, confidence=0.99):
    dated_frame(returns)
    w = allocation(weights, returns.shape[1])
    if window < 20 or len(returns) <= window + 1:
        raise ValueError("Rolling window must leave at least two out-of-sample observations.")
    rows = []
    for t in range(window, len(returns)):
        estimate = engine.estimate(returns.iloc[t - window : t], w, confidence, horizon=1)
        loss = -float(returns.iloc[t].to_numpy() @ w)
        rows.append(
            {
                "date": returns.index[t],
                "var": estimate.var,
                "expected_shortfall": estimate.expected_shortfall,
                "realized_loss": loss,
                "exception": bool(loss > estimate.var),
            }
        )
    forecasts = pd.DataFrame(rows).set_index("date")
    return forecasts, coverage_tests(forecasts.exception, confidence)
