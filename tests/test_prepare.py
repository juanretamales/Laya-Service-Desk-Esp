"""Tests unitarios para el preparador estructural básico."""

import pytest
from src.preprocessing.prepare import normalize, run


def test_normalize():
    """Valida la normalización de columnas esperadas."""
    m = {k: k for k in ("ticket_id", "subject", "description", "category", "priority", "support_group")}
    row = {k: "x" for k in m}
    r = normalize(row, m, "src")
    assert r["source_id"] == "src"


def test_missing_column():
    """Valida que lance error si faltan columnas requeridas."""
    with pytest.raises(ValueError):
        normalize({"ticket_id": "1"}, {}, "src")


def test_disabled_sources(tmp_path):
    """Valida que lance error si no hay fuentes habilitadas."""
    p = tmp_path / "c.yaml"
    p.write_text("sources: [{id: s, enabled: false}]", encoding="utf-8")
    with pytest.raises(ValueError):
        run(str(p), str(tmp_path / "out.jsonl"))
