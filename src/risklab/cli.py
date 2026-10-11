from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from risklab.data import synthetic_market, load_returns, ASSET_CLASSES, file_hash
from risklab.engines import HistoricalEngine, ParametricEngine, MonteCarloEngine
from risklab.analytics import portfolio_statistics, pca_attribution
from risklab.stress import load_scenarios
from risklab.backtesting import rolling_var
from risklab.optimization import RiskParityOptimizer, EqualWeightAllocator
from risklab.regimes import walk_forward_risk_parity


def run(returns, asset_classes, output, scenario_path, confidence=0.99, window=252, draws=6000, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(asset_classes) != list(returns.columns):
        raise ValueError("Asset-class mapping must match ordered return columns.")
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    engines = {
        "historical": HistoricalEngine(),
        "parametric": ParametricEngine(),
        "monte_carlo": MonteCarloEngine(draws=draws),
    }
    report = {
        "source": source or {"type": "synthetic", "seed": 42},
        "rows": len(returns),
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "asset_classes": asset_classes,
        "baseline_weights": dict(zip(returns.columns, map(float, weights))),
        "confidence": confidence,
        "risk_estimates": {
            name: asdict(engine.estimate(returns.iloc[-window:], weights, confidence))
            for name, engine in engines.items()
        },
        "student_t_monte_carlo": asdict(
            MonteCarloEngine(draws=draws, distribution="student_t").estimate(
                returns.iloc[-window:], weights, confidence
            )
        ),
    }
    report["portfolio_statistics"], drawdown = portfolio_statistics(returns, weights)
    report["pca_attribution"] = pca_attribution(returns, weights, list(returns.columns))
    report["stress_tests"] = [
        scenario.evaluate(asset_classes, weights) for scenario in load_scenarios(scenario_path)
    ]
    report["var_backtests"] = {}
    forecasts = {}
    for name, engine in engines.items():
        forecast, tests = rolling_var(returns, weights, engine, window, confidence)
        forecast.to_csv(output / f"{name}_forecasts.csv")
        report["var_backtests"][name] = tests
        forecasts[name] = forecast
    optimizer = RiskParityOptimizer(upper_bound=max(0.6, 1 / returns.shape[1]), turnover_cap=0.8)
    optimized, rebalances = walk_forward_risk_parity(returns, optimizer, window)
    baseline, _ = walk_forward_risk_parity(returns, EqualWeightAllocator(), window)
    baseline.to_csv(output / "equal_weight_returns.csv")
    optimized.to_csv(output / "risk_parity_returns.csv")
    (output / "rebalances.json").write_text(json.dumps(rebalances, indent=2))
    report["risk_parity"] = {
        "rebalances": len(rebalances),
        "total_traded_notional_multiple": float(sum(r["turnover"] for r in rebalances)),
        "net_return_statistics": portfolio_statistics(optimized[["net_return"]], [1])[0],
        "equal_weight_net_statistics": portfolio_statistics(baseline[["net_return"]], [1])[0],
        "mean_risk_budget_error": float(np.mean([r["risk_budget_squared_error"] for r in rebalances])),
        "regime_rebalance_counts": pd.Series([r["regime"] for r in rebalances]).value_counts().to_dict(),
    }
    report["limitations"] = [
        "One-day backtests; historical tail estimates have sparse observations at 99%.",
        "Static linear P&L risk models; Gaussian assumptions can fail in fat-tailed markets.",
        "Stress shocks are hypothetical asset-class return shocks, not derivative repricing.",
        "Risk parity minimizes risk-budget error under constraints; exact parity is not guaranteed.",
        "Regime labels use trailing data and fixed thresholds, not future returns.",
        "Research analysis, not a production capital calculation or investment recommendation.",
    ]
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    from risklab.plotting import figures

    figures(returns, forecasts, optimized, baseline, rebalances, report, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "backtests": report["var_backtests"],
                "risk_parity": report["risk_parity"],
            },
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="RiskLab portfolio risk analytics")
    parser.add_argument("--input", help="CSV with Date index and one column per asset")
    parser.add_argument(
        "--prices", action="store_true", help="Convert complete positive price levels to simple returns"
    )
    parser.add_argument("--asset-classes", help="JSON mapping ordered columns to equity/bond/commodity/fx")
    parser.add_argument("--output", default="reports/synthetic")
    parser.add_argument("--scenarios", default="config/scenarios.json")
    parser.add_argument("--rows", type=int, default=1512)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--draws", type=int, default=6000)
    args = parser.parse_args()
    if args.input:
        returns = load_returns(args.input, args.prices)
        if not args.asset_classes:
            parser.error("--asset-classes required for external CSV data")
        classes = json.loads(Path(args.asset_classes).read_text())
        source = {
            "type": "external CSV",
            "sha256": file_hash(args.input),
            "path_basename": Path(args.input).name,
            "input_prices": args.prices,
        }
        provenance = Path(args.input).parent / "source.json"
        if provenance.exists():
            source["metadata"] = json.loads(provenance.read_text())
    else:
        returns, _ = synthetic_market(args.rows)
        classes = ASSET_CLASSES
        source = {"type": "synthetic", "seed": 42}
    run(returns, classes, args.output, args.scenarios, args.confidence, args.window, args.draws, source)


if __name__ == "__main__":
    main()
