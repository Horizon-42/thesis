from __future__ import annotations

from pathlib import Path

import pytest

from faa_fix_lookup import (
    decode_cifp_coordinate,
    faacifp_path,
    lookup_local_cifp_fix,
)


CIFP_ROOT = Path(__file__).parents[3] / "data" / "CIFP" / "CIFP_260319"


def test_decode_cifp_coordinate() -> None:
    assert decode_cifp_coordinate("N36030596") == pytest.approx(36.0516556)
    assert decode_cifp_coordinate("W078500425") == pytest.approx(-78.8345139)


def test_lookup_local_cifp_fix_resolves_duham() -> None:
    result = lookup_local_cifp_fix("DUHAM", faacifp_path(CIFP_ROOT))

    assert result is not None
    assert result.ident == "DUHAM"
    assert result.lat == pytest.approx(36.0516556)
    assert result.lon == pytest.approx(-78.8345139)
    assert result.source == "local-cifp"


def test_lookup_local_cifp_fix_resolves_kasle() -> None:
    result = lookup_local_cifp_fix("KASLE", faacifp_path(CIFP_ROOT))

    assert result is not None
    assert result.ident == "KASLE"
    assert result.lat == pytest.approx(35.905925)
    assert result.lon == pytest.approx(-78.8204556)
