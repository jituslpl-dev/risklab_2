import json
from pathlib import Path
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from risklab.validation import allocation


@dataclass(frozen=True)
class StressScenario:
    name: str
    class_shocks: dict[str, float]

    def __post_init__(self):
        object.__setattr__(self, "class_shocks", MappingProxyType(dict(self.class_shocks)))

    def evaluate(self, asset_classes, weights, notional=1_000_000):
        names = list(asset_classes)
        w = allocation(weights, len(names))
        if not np.isfinite(notional) or notional <= 0:
            raise ValueError("Notional must be finite and positive.")
        shocks = []
        for name in names:
            value = self.class_shocks[asset_classes[name]]
            if not np.isfinite(value) or value <= -1:
                raise ValueError("Stress returns must be finite and above -100%.")
            shocks.append(value)
        contributions = w * np.asarray(shocks)
        return {
            "scenario": self.name,
            "portfolio_return": float(contributions.sum()),
            "pnl": float(contributions.sum() * notional),
            "asset_return_contributions": dict(zip(names, map(float, contributions))),
        }


def load_scenarios(path):
    content = json.loads(Path(path).read_text())
    if len(content) != 4:
        raise ValueError("The configured scenario suite must contain four scenarios.")
    return [StressScenario(name, shocks) for name, shocks in content.items()]
