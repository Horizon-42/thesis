"""One strategy per prediction path (package review §4.2, 2026-09-10).

The spine — `dataset`, `batching`, `objective`, `forecast`, `validation`, `export`,
`train`, `models` — used to branch on ``prediction_output`` at nine places, each an
if/elif over the three paths, and each path's code sat in the spine module that called
it. Each branch is now a method of that path's strategy (`base.OutputStrategy`), the
path's code sits in its own package here, and the spine calls
``strategy(config).<method>``. A fourth path (the plan-and-guidance design) is one
package under ``outputs/`` with one strategy class, and no edit in the spine.

What more than one path consumes lives here rather than under a path (the membership
rule, 2026-09-10): ``dynamics/`` (the flight models, the rollout API, the inverses, the
command-hook contract), ``constraints/`` (the command hooks), ``envelope`` (the
dimensionless command box and its newton conversion), ``conditioning`` (the condition
vector) and ``duration_heads``.

The registry is lazy on purpose: the strategies import the spine's shared value types
(`Forecast`, `FlightSeries`, `LossComponents`), and the spine imports this module, so a
strategy module is loaded on first use — after every spine module is initialised.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from ts_transformer.config import (
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
    PREDICTION_OUTPUTS,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.outputs.base import (
    ForecastOptions,
    OutputStrategy,
    Replay,
    WindowContext,
)

_STRATEGY_MODULES: dict[str, str] = {
    PREDICTION_STATE: "ts_transformer.outputs.state.strategy",
    PREDICTION_CLOSURE: "ts_transformer.outputs.closure.strategy",
    PREDICTION_CONTROL: "ts_transformer.outputs.control.strategy",
}
if set(_STRATEGY_MODULES) != set(PREDICTION_OUTPUTS):
    raise RuntimeError("every prediction_output needs a strategy module")


@lru_cache(maxsize=None)
def strategy_for(prediction_output: str) -> OutputStrategy:
    """The strategy of one ``prediction_output`` (a stored forecast's, a config's)."""
    try:
        module = _STRATEGY_MODULES[prediction_output]
    except KeyError:
        raise ValueError(
            f"unknown prediction_output {prediction_output!r}; expected one of "
            f"{PREDICTION_OUTPUTS}"
        ) from None
    return importlib.import_module(module).STRATEGY


def strategy(config: TSConfig) -> OutputStrategy:
    """The strategy a run's config selects."""
    return strategy_for(config.prediction_output)


__all__ = [
    "ForecastOptions",
    "OutputStrategy",
    "Replay",
    "WindowContext",
    "strategy",
    "strategy_for",
]
