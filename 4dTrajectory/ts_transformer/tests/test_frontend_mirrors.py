"""The frontend's mirrors of this package's vocabularies, pinned.

`aeroviz-4d/src/data/airportData.ts` restates two of this package's vocabularies in
TypeScript — `EXPERIMENT_PREDICTION_OUTPUTS` for `config.PREDICTION_OUTPUTS` and
`EXPERIMENT_HORIZON_MODES` for `config.HORIZON_MODES`. It cannot import them, the publisher
writes each run's values straight through, and the picker's manifest validator rejects a
category whose value is not listed — and, through `.every`, the whole airport's manifest
with it. Twice a new output was published before the mirror learned it (`closure`
2026-09-07, `plan` 2026-09-12); each time every airport's picker went empty. These tests
fail the day a mirror differs from its source. The comparison is ORDER-SENSITIVE on
purpose: the mirror is meant to read like the source, and a reorder is a one-line edit.
"""

from __future__ import annotations

import re

import pytest

from ts_transformer.config import HORIZON_MODES, PREDICTION_OUTPUTS
from ts_transformer.repo_layout import REPO_ROOT

AIRPORT_DATA_TS = REPO_ROOT / "aeroviz-4d" / "src" / "data" / "airportData.ts"

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def frontend_vocabulary(name: str) -> tuple[str, ...]:
    """The string literals of `export const <name> = [...] as const` in the frontend file."""
    source = AIRPORT_DATA_TS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name} = \[(?P<body>.*?)\]\s*as\s+const", source, re.DOTALL)
    assert match is not None, f"{name} not found in {AIRPORT_DATA_TS}"
    return tuple(re.findall(r'"([^"]+)"', _COMMENT.sub("", match.group("body"))))


@pytest.mark.parametrize(
    ("frontend_name", "vocabulary"),
    [
        ("EXPERIMENT_PREDICTION_OUTPUTS", PREDICTION_OUTPUTS),
        ("EXPERIMENT_HORIZON_MODES", HORIZON_MODES),
    ],
)
def test_frontend_vocabulary_mirrors_the_config(frontend_name, vocabulary):
    assert frontend_vocabulary(frontend_name) == tuple(vocabulary)
