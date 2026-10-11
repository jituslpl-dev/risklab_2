import numpy as np
import pandas as pd
from risklab.optimization import RiskParityOptimizer
from risklab.validation import dated_frame


def classify_regime(training):
    # Fixed thresholds, not future/full-sample quantiles. Equal-weight market proxy.
    volatility = float(training.iloc[-20:].mean(axis=1).std(ddof=1) * np.sqrt(252))
    regime = "calm" if volatility < 0.10 else ("high_volatility" if volatility < 0.20 else "crisis")
    return regime, volatility


def walk_forward_risk_parity(returns, optimizer=None, window=252, rebalance_every=21, cost_bps=5):
    dated_frame(returns)
    if window < 20 or len(returns) <= window or rebalance_every < 1 or cost_bps < 0 or cost_bps > 100:
        raise ValueError("Invalid walk-forward configuration.")
    optimizer = optimizer or RiskParityOptimizer()
    drifted = None
    records = []
    daily = []
    for t in range(window, len(returns)):
        transaction_cost = 0.0
        if (t - window) % rebalance_every == 0:
            history = returns.iloc[t - window : t]
            solved = optimizer.solve(history, previous=drifted)
            target = np.asarray(solved["weights"])
            turnover = float(np.abs(target - (np.zeros(len(target)) if drifted is None else drifted)).sum())
            transaction_cost = turnover * cost_bps / 10000
            regime, vol = classify_regime(history)
            records.append(
                {
                    **solved,
                    "date": str(returns.index[t].date()),
                    "regime": regime,
                    "proxy_volatility": vol,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                }
            )
            drifted = target
        asset_return = returns.iloc[t].to_numpy()
        gross = float(drifted @ asset_return)
        net = (1 - transaction_cost) * (1 + gross) - 1
        daily.append(
            {
                "date": returns.index[t],
                "gross_return": gross,
                "net_return": net,
                "transaction_cost": transaction_cost,
            }
        )
        drifted = drifted * (1 + asset_return) / (1 + gross)
    return pd.DataFrame(daily).set_index("date"), records
