"""
Pruebas unitarias para el módulo de unificación y mapeo semántico de etiquetas (src/preprocessing/unify.py).
"""

import json
from pathlib import Path
import pytest

from src.preprocessing.unify import (
    CANONICAL_CATEGORIES,
    CANONICAL_PRIORITIES,
    CANONICAL_SUPPORT_GROUPS,
    PRIORITY_MAP,
    SUPPORT_GROUP_MAP,
    ZENODO_CATEGORY_MAP,
    build_unified_record,
    export_unified_jsonl,
    load_zenodo_records,
    map_canonical_label,
)


def test_map_canonical_label_exact():
    """Verifica que etiquetas que ya son canónicas se mantengan inalteradas."""
    for cat in ("access", "network", "hardware", "software", "email", "database", "other"):
        assert map_canonical_label(cat, ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == cat


def test_map_canonical_label_zenodo_semantic():
    """Verifica el mapeo semántico de las 7 clases originales de Zenodo."""
    assert map_canonical_label("Active Directory", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "access"
    assert map_canonical_label("Fileservice", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "software"
    assert map_canonical_label("Software", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "software"
    assert map_canonical_label("O365", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "email"
    assert map_canonical_label("Computer-Services", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "hardware"
    assert map_canonical_label("EOL", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "hardware"
    assert map_canonical_label("Support general", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "other"


def test_map_canonical_label_unknown_fallback():
    """Verifica que valores desconocidos, vacíos o no inferibles mapeen a 'unknown'."""
    assert map_canonical_label("clase_inexistente", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "unknown"
    assert map_canonical_label("", ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "unknown"
    assert map_canonical_label(None, ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES) == "unknown"


def test_map_priority_and_support_groups():
    """Verifica los diccionarios de prioridades y grupos resolutores."""
    assert map_canonical_label("Baja", PRIORITY_MAP, CANONICAL_PRIORITIES) == "low"
    assert map_canonical_label("Urgente", PRIORITY_MAP, CANONICAL_PRIORITIES) == "critical"
    assert map_canonical_label("Redes", SUPPORT_GROUP_MAP, CANONICAL_SUPPORT_GROUPS) == "network"
    assert map_canonical_label("Mesa de ayuda", SUPPORT_GROUP_MAP, CANONICAL_SUPPORT_GROUPS) == "service_desk"


def test_build_unified_record_valid():
    """Verifica la construcción de un registro con sanitización PII e inferencia de idioma."""
    rec = build_unified_record(
        ticket_id="TCK-999",
        subject="Fallo en la VPN para juan@corp.com",
        description="No puedo acceder a la IP 192.168.1.1 desde mi casa.",
        category="network",
        priority="high",
        support_group="network",
        source_id="synthetic_test",
        anonymize=True,
    )

    assert rec["ticket_id"] == "TCK-999"
    assert rec["category"] == "network"
    assert rec["priority"] == "high"
    assert rec["support_group"] == "network"
    assert rec["language"] == "es"
    assert "<EMAIL>" in rec["subject"]
    assert "<IP>" in rec["description"]
    assert "juan@corp.com" not in rec["subject"]


def test_build_unified_record_validation_errors():
    """Verifica que se rechacen tickets sin ID o sin ningún contenido textual."""
    with pytest.raises(ValueError, match="ticket_id vacío"):
        build_unified_record("", "Asunto", "Texto", "other", "low", "other", "test")

    with pytest.raises(ValueError, match="sin asunto ni descripción"):
        build_unified_record("TCK-01", "", "", "other", "low", "other", "test")


def test_load_zenodo_records_mock(tmp_path: Path):
    """Verifica la carga y unión de archivos X_train y y_train de Zenodo."""
    x_file = tmp_path / "X_train.csv"
    y_file = tmp_path / "y_train.csv"

    x_file.write_text("id,text\n10,Error en acceso O365\n20,Fallo en servidor LVS\n", encoding="utf-8")
    y_file.write_text("id,category_truth\n10,O365\n20,Fileservice\n", encoding="utf-8")

    records = load_zenodo_records(tmp_path, anonymize=True)
    assert len(records) == 2

    r10 = next(r for r in records if r["ticket_id"] == "zenodo_10")
    assert r10["category"] == "email"  # O365 -> email
    assert r10["priority"] == "unknown"

    r20 = next(r for r in records if r["ticket_id"] == "zenodo_20")
    assert r20["category"] == "software"  # Fileservice -> software


def test_export_unified_jsonl(tmp_path: Path):
    """Verifica la exportación correcta y la salvaguarda de sobreescritura de JSONL."""
    out_file = tmp_path / "unified.jsonl"
    sample_records = [
        {
            "ticket_id": "1",
            "subject": "s1",
            "description": "d1",
            "category": "software",
            "priority": "low",
            "support_group": "service_desk",
            "source_id": "test",
            "language": "es",
        }
    ]

    export_unified_jsonl(sample_records, out_file)
    assert out_file.exists()

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["ticket_id"] == "1"

    # Verificación de protección contra sobreescritura sin overwrite=True
    with pytest.raises(FileExistsError, match="El archivo unificado ya existe"):
        export_unified_jsonl(sample_records, out_file, overwrite=False)

    # Con overwrite=True debe sobreescribir sin error
    export_unified_jsonl(sample_records, out_file, overwrite=True)
