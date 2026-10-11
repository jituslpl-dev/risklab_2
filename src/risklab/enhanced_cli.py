"""Compose preserved RiskLab workflow with independently reportable enhancements."""

from pathlib import Path
from dataclasses import asdict
import argparse
import html
import json
import numpy as np
from risklab.cli import run
from risklab.data import synthetic_market, load_returns, ASSET_CLASSES, file_hash
from risklab.advanced import (
    FilteredHistoricalEngine,
    BlockBootstrapEngine,
    expected_shortfall_attribution,
    bootstrap_tail_intervals,
)
from risklab.engines import MonteCarloEngine
from risklab.backtesting import rolling_var
from risklab.diagnostics import forecast_diagnostics
from risklab.tail_optimization import CVaROptimizer, reverse_stress
from risklab.regimes import walk_forward_risk_parity
from risklab.analytics import portfolio_statistics
from risklab.hierarchical import HierarchicalRiskParity


def enhanced_run(
    returns,
    classes,
    output,
    scenarios="config/scenarios.json",
    window=252,
    draws=6000,
    confidence=0.99,
    source=None,
):
    output = Path(output)
    base = run(returns, classes, output, scenarios, confidence, window, draws, source)
    w = np.ones(returns.shape[1]) / returns.shape[1]
    history = returns.iloc[-window:]
    engines = {
        "filtered_historical": FilteredHistoricalEngine(),
        "block_bootstrap": BlockBootstrapEngine(draws=draws),
        "student_t": MonteCarloEngine(draws=draws, distribution="student_t"),
    }
    report = {
        "schema_version": 2,
        "source": base["source"],
        "configuration": {
            "window": window,
            "draws": draws,
            "confidence": confidence,
            "transaction_cost_bps": 5,
            "rebalance_every": 21,
        },
        "estimates": {
            name: asdict(engine.estimate(history, w, confidence)) for name, engine in engines.items()
        },
        "ten_day_block_bootstrap": asdict(engines["block_bootstrap"].estimate(history, w, confidence, 10)),
        "es_attribution": expected_shortfall_attribution(history, w, confidence),
        "tail_sampling_intervals": bootstrap_tail_intervals(history, w, confidence),
        "reverse_stress": reverse_stress(history, w),
        "forecast_comparison": {},
    }
    import pandas as pd

    for name in base["var_backtests"]:
        frame = pd.read_csv(output / f"{name}_forecasts.csv")
        report["forecast_comparison"][name] = forecast_diagnostics(frame, confidence)
    for name, engine in engines.items():
        forecasts, _ = rolling_var(returns, w, engine, window, confidence)
        forecasts.to_csv(output / f"{name}_forecasts.csv")
        report["forecast_comparison"][name] = forecast_diagnostics(forecasts, confidence)
    optimizer = CVaROptimizer(upper_bound=max(0.6, 1 / len(w)))
    daily, rebalances = walk_forward_risk_parity(returns, optimizer, window)
    daily.to_csv(output / "cvar_returns.csv")
    (output / "cvar_rebalances.json").write_text(json.dumps(rebalances, indent=2))
    report["cvar_strategy"] = {
        "objective_confidence": 0.95,
        "statistics": portfolio_statistics(daily[["net_return"]], [1])[0],
        "rebalances": len(rebalances),
        "total_turnover": sum(x["turnover"] for x in rebalances),
    }
    report["latest_cvar_solution"] = optimizer.solve(history)
    hrp_daily, hrp_records = walk_forward_risk_parity(returns, HierarchicalRiskParity(), window)
    hrp_daily.to_csv(output / "hrp_returns.csv")
    (output / "hrp_rebalances.json").write_text(json.dumps(hrp_records, indent=2))
    report["hrp_strategy"] = {
        "statistics": portfolio_statistics(hrp_daily[["net_return"]], [1])[0],
        "rebalances": len(hrp_records),
        "total_turnover": sum(x["turnover"] for x in hrp_records),
        "constraint_note": "Long-only and fully invested; original risk-parity bounds and turnover caps do not apply to HRP.",
    }
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, filename in [
        ("Equal weight", "equal_weight_returns.csv"),
        ("Constrained risk parity", "risk_parity_returns.csv"),
        ("Constrained CVaR", "cvar_returns.csv"),
        ("Hierarchical risk parity", "hrp_returns.csv"),
    ]:
        frame = pd.read_csv(output / filename, index_col=0, parse_dates=True)
        ax.plot(frame.index, (1 + frame.net_return).cumprod(), label=name)
    ax.set(
        title="Four walk-forward allocations with matching timing and 5 bps trading costs",
        ylabel="Net wealth from 1.0",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "enhanced_wealth.png", dpi=150)
    plt.close(fig)
    (output / "enhancements.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest = {p.name: file_hash(p) for p in sorted(output.glob("*.csv"))}
    manifest.update(
        {p.name: file_hash(p) for p in sorted(output.glob("*.json")) if p.name != "manifest.json"}
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    dashboard(output, base, report)
    return report


def dashboard(output, base, enhanced):
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{r['exceptions']}</td><td>{r['exception_rate']:.2%}</td>"
        f"<td>{r['kupiec_pvalue']:.4g}</td><td>{r['mean_quantile_loss']:.6f}</td></tr>"
        for name, r in enhanced["forecast_comparison"].items()
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RiskLab 2 research report</title><style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17334b;background:#f4f7fa}}
table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:white;padding:20px}}</style>
<h1>RiskLab 2 · Portfolio risk research</h1><p>{base["start"]} to {base["end"]} · {base["rows"]} observations · {base["confidence"]:.0%} VaR</p>
<p>Measured research results. Cross-asset validation is synthetic; public sample is historical equities only.</p>
<h2>Rolling forecast comparison</h2><table><tr><th>Model</th><th>Exceptions</th><th>Rate</th><th>Coverage p</th><th>Quantile loss</th></tr>{rows}</table>
<p>Lower quantile loss is preferable on this shared sample. Coverage non-rejection does not establish correctness. No model is selected using these test results.</p>
<h2>Four strategy comparison</h2><img alt="Four walk-forward strategies" src="figures/enhanced_wealth.png">
<h2>Tail attribution and reverse stress</h2><pre>{html.escape(json.dumps({"ES attribution": enhanced["es_attribution"], "reverse stress": enhanced["reverse_stress"]}, indent=2))}</pre>
<p>See summary.json and enhancements.json for costs, constraints, full strategy results and uncertainty.</p></html>"""
    (output / "dashboard.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="RiskLab 2 enhanced portfolio research")
    parser.add_argument("--input")
    parser.add_argument("--prices", action="store_true")
    parser.add_argument("--asset-classes")
    parser.add_argument("--rows", type=int, default=1512)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--draws", type=int, default=6000)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--output", default="reports/synthetic")
    parser.add_argument("--scenarios", default="config/scenarios.json")
    args = parser.parse_args()
    if args.window < 40:
        parser.error("Enhanced models require a window of at least 40.")
    if args.input:
        if not args.asset_classes:
            parser.error("External data requires --asset-classes.")
        returns = load_returns(args.input, args.prices)
        classes = json.loads(Path(args.asset_classes).read_text())
        source = {"type": "external CSV", "sha256": file_hash(args.input), "input_prices": args.prices}
        provenance = Path(args.input).parent / "source.json"
        if provenance.exists():
            source["metadata"] = json.loads(provenance.read_text())
    else:
        returns, _ = synthetic_market(args.rows)
        classes = ASSET_CLASSES
        source = {"type": "synthetic", "seed": 42}
    enhanced_run(
        returns, classes, args.output, args.scenarios, args.window, args.draws, args.confidence, source
    )


if __name__ == "__main__":
    main()
