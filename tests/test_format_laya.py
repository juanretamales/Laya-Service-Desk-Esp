"""Tests unitarios para el formateo de datos adaptado a Laya Router."""

import json
from pathlib import Path

from src.preprocessing.format_laya import (
    build_default_laya_questions,
    export_laya_formatted_dataset,
    format_record_for_laya,
)


def test_build_default_laya_questions():
    """Valida que las preguntas por defecto contengan las preguntas y tipos requeridos."""
    questions = build_default_laya_questions()
    assert "category" in questions
    assert "priority" in questions
    assert "support_group" in questions

    for key, q_data in questions.items():
        assert "question" in q_data
        assert q_data["type"] == "choice"
        assert isinstance(q_data["options"], list)
        assert len(q_data["options"]) > 0


def test_format_record_for_laya():
    """Valida la transformación de un registro canónico a la estructura de Laya."""
    record = {
        "ticket_id": "test_001",
        "source_id": "synthetic",
        "subject": "Fallo en impresora",
        "description": "No imprime en red local",
        "category": "hardware",
        "priority": "medium",
        "support_group": "workplace",
        "language": "es",
        "content_hash": "abc123hash",
    }

    formatted = format_record_for_laya(record)

    assert formatted["ticket_id"] == "test_001"
    assert formatted["source_id"] == "synthetic"
    assert formatted["language"] == "es"
    assert formatted["content_hash"] == "abc123hash"
    assert formatted["state"]["subject"] == "Fallo en impresora"
    assert formatted["state"]["body"] == "No imprime en red local"
    assert formatted["targets"]["category"] == "hardware"
    assert formatted["targets"]["priority"] == "medium"
    assert formatted["targets"]["support_group"] == "workplace"
    assert "category" in formatted["questions"]


def test_format_record_missing_subject():
    """Valida el manejo robusto de registros sin asunto especificado."""
    record = {
        "ticket_id": "test_002",
        "source_id": "zenodo",
        "subject": "",
        "description": "Solo cuerpo de ticket",
        "category": "software",
        "priority": "unknown",
        "support_group": "unknown",
    }

    formatted = format_record_for_laya(record)
    assert formatted["state"]["subject"] == "Sin asunto especificado"
    assert formatted["state"]["body"] == "Solo cuerpo de ticket"


def test_export_laya_formatted_dataset(tmp_path: Path):
    """Valida la exportación en disco de un conjunto de datos formateado para Laya."""
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output_laya.jsonl"

    sample_records = [
        {
            "ticket_id": "t1",
            "subject": "Acceso VPN",
            "description": "Error al conectar",
            "category": "network",
            "priority": "high",
            "support_group": "network",
            "language": "es",
            "content_hash": "hash1",
        },
        {
            "ticket_id": "t2",
            "subject": "Printer issue",
            "description": "Paper jam",
            "category": "hardware",
            "priority": "low",
            "support_group": "workplace",
            "language": "en",
            "content_hash": "hash2",
        },
    ]

    with input_file.open("w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")

    # Exportar todos
    count = export_laya_formatted_dataset(input_file, output_file)
    assert count == 2
    assert output_file.exists()

    # Exportar con filtro de idioma 'es'
    output_es = tmp_path / "output_es.jsonl"
    count_es = export_laya_formatted_dataset(input_file, output_es, filter_language="es")
    assert count_es == 1

    lines = [json.loads(line) for line in output_es.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["language"] == "es"
