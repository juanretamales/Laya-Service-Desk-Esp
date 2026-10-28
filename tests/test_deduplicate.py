"""
Pruebas unitarias para el módulo de deduplicación de tickets (src/preprocessing/deduplicate.py).
"""

import json
from pathlib import Path
import pytest

from src.preprocessing.deduplicate import (
    compute_content_hash,
    deduplicate_records,
    export_deduplication_results,
    normalize_text_for_hashing,
)


def test_normalize_text_for_hashing():
    """Verifica la normalización de espacios, puntuación y mayúsculas/minúsculas."""
    raw1 = "  Fallo crítico en VPN corporativa...  "
    raw2 = "fallo crítico en vpn corporativa"
    assert normalize_text_for_hashing(raw1) == raw2
    assert normalize_text_for_hashing("") == ""


def test_compute_content_hash_invariance():
    """Verifica que variaciones de espacios o puntuación produzcan el mismo hash SHA-256."""
    rec1 = {
        "subject": "Fallo de conexión VPN",
        "description": "No conecta a la red corporativa.",
    }
    rec2 = {
        "subject": "  fallo de conexión vpn!  ",
        "description": "No conecta a la red corporativa...",
    }
    rec3 = {
        "subject": "Fallo de conexión VPN",
        "description": "Otro problema completamente diferente.",
    }

    hash1 = compute_content_hash(rec1)
    hash2 = compute_content_hash(rec2)
    hash3 = compute_content_hash(rec3)

    assert hash1 == hash2
    assert hash1 != hash3


def test_deduplicate_records():
    """Verifica la segregación exacta entre registros únicos y duplicados."""
    records = [
        {
            "ticket_id": "T1",
            "subject": "Problema con mouse",
            "description": "El puntero no se mueve en pantalla.",
            "source_id": "zenodo",
        },
        {
            "ticket_id": "T2",
            "subject": "  problema con mouse!  ",
            "description": "El puntero no se mueve en pantalla...",
            "source_id": "zenodo",
        },
        {
            "ticket_id": "T3",
            "subject": "Acceso a base de datos",
            "description": "Solicito usuario para Oracle.",
            "source_id": "kaggle",
        },
        {
            "ticket_id": "T4",
            "subject": "Problema con mouse",
            "description": "El puntero no se mueve en pantalla.",
            "source_id": "kaggle",
        },
    ]

    unique, duplicates, stats = deduplicate_records(records)

    assert len(unique) == 2
    assert len(duplicates) == 2

    unique_ids = [r["ticket_id"] for r in unique]
    assert unique_ids == ["T1", "T3"]

    dup_t2 = next(r for r in duplicates if r["ticket_id"] == "T2")
    assert dup_t2["duplicate_of_ticket_id"] == "T1"

    dup_t4 = next(r for r in duplicates if r["ticket_id"] == "T4")
    assert dup_t4["duplicate_of_ticket_id"] == "T1"

    assert stats["total_records"] == 4
    assert stats["unique_records"] == 2
    assert stats["duplicate_records"] == 2
    assert stats["duplicate_ratio"] == 0.5
    assert stats["duplicates_by_source"] == {"zenodo": 1, "kaggle": 1}


def test_export_deduplication_results(tmp_path: Path):
    """Verifica la persistencia de registros limpios, duplicados y estadísticas JSON."""
    clean_out = tmp_path / "clean.jsonl"
    dup_out = tmp_path / "dup.jsonl"
    stats_out = tmp_path / "stats.json"

    records = [
        {"ticket_id": "1", "subject": "A", "description": "B", "source_id": "src1"},
        {"ticket_id": "2", "subject": "A", "description": "B", "source_id": "src1"},
    ]
    unique, duplicates, stats = deduplicate_records(records)

    export_deduplication_results(
        unique_records=unique,
        duplicate_records=duplicates,
        stats=stats,
        output_clean_jsonl=clean_out,
        output_duplicates_jsonl=dup_out,
        output_stats_json=stats_out,
    )

    assert clean_out.exists()
    assert dup_out.exists()
    assert stats_out.exists()

    clean_lines = [json.loads(line) for line in clean_out.read_text(encoding="utf-8").strip().split("\n")]
    assert len(clean_lines) == 1
    assert clean_lines[0]["ticket_id"] == "1"

    stats_data = json.loads(stats_out.read_text(encoding="utf-8"))
    assert stats_data["duplicate_records"] == 1

    # Comprobar protección contra sobreescritura sin overwrite=True
    with pytest.raises(FileExistsError, match="El archivo destino ya existe"):
        export_deduplication_results(
            unique_records=unique,
            duplicate_records=duplicates,
            stats=stats,
            output_clean_jsonl=clean_out,
            overwrite=False,
        )
