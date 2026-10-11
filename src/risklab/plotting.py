from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figures(returns, forecasts, optimized, baseline, rebalances, report, output):
    output = Path(output) / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"figure.figsize": (11, 5), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    fig, ax = plt.subplots()
    for name, forecast in forecasts.items():
        ax.plot(forecast.index, forecast["var"], label=name, alpha=0.8)
    first = next(iter(forecasts.values()))
    ax.plot(first.index, first.realized_loss, color="gray", alpha=0.3, lw=0.6, label="realized loss")
    ax.set(
        title="Rolling one-day VaR forecasts (strictly prior windows)",
        ylabel="Fraction of portfolio notional",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rolling_var.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(
        baseline.index,
        (1 + baseline.net_return).cumprod(),
        label="equal weight, monthly rebalanced, net of costs",
    )
    ax.plot(
        optimized.index,
        (1 + optimized.net_return).cumprod(),
        label="risk parity, monthly rebalanced, net of costs",
    )
    ax.set(title="Walk-forward wealth: matching rebalancing and cost conventions", ylabel="Wealth from 1.0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "wealth.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    values = np.array([r["weights"] for r in rebalances])
    dates = [r["date"] for r in rebalances]
    ax.stackplot(np.arange(len(dates)), values.T, labels=returns.columns)
    ax.set(
        title="Constrained risk-parity allocation at each rebalance",
        xlabel="Rebalance number",
        ylabel="Weight",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "allocations.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    shocks = report["stress_tests"]
    ax.barh([r["scenario"] for r in shocks], [r["portfolio_return"] * 100 for r in shocks], color="#b45309")
    ax.set(title="Four hypothetical asset-class stress scenarios", xlabel="Portfolio return (%)")
    fig.tight_layout()
    fig.savefig(output / "stress.png", dpi=150)
    plt.close(fig)
