"""Formateo de tickets al contrato de entrada de Laya Router.

Este módulo transforma los registros canónicos deduplicados y unificados
al formato de preguntas tipadas ('choice', 'score') y estado contextual
requerido por laya.Router.predict(state, questions).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

from src.preprocessing.unify import (
    CANONICAL_CATEGORIES,
    CANONICAL_PRIORITIES,
    CANONICAL_SUPPORT_GROUPS,
)


def build_default_laya_questions() -> Dict[str, Dict[str, Any]]:
    """Construye el conjunto canónico de preguntas tipadas para Laya Router.

    Returns:
        Diccionario con preguntas tipadas estructuradas para laya.Router.
    """
    return {
        "category": {
            "type": "choice",
            "instructions": "¿Cuál es la categoría canónica correspondiente a este incidente o requerimiento?",
            "criteria": list(CANONICAL_CATEGORIES),
            "question": "¿Cuál es la categoría canónica correspondiente a este incidente o requerimiento?",
            "options": list(CANONICAL_CATEGORIES),
        },
        "priority": {
            "type": "choice",
            "instructions": "¿Cuál es el nivel de prioridad operativa para este ticket de soporte?",
            "criteria": list(CANONICAL_PRIORITIES),
            "question": "¿Cuál es el nivel de prioridad operativa para este ticket de soporte?",
            "options": list(CANONICAL_PRIORITIES),
        },
        "support_group": {
            "type": "choice",
            "instructions": "¿Qué grupo de soporte especializado debe gestionar y resolver este ticket?",
            "criteria": list(CANONICAL_SUPPORT_GROUPS),
            "question": "¿Qué grupo de soporte especializado debe gestionar y resolver este ticket?",
            "options": list(CANONICAL_SUPPORT_GROUPS),
        },
    }


def format_record_for_laya(
    record: Dict[str, Any],
    questions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Adapta un registro canónico al formato dual estado/preguntas de Laya.

    Args:
        record: Diccionario canónico con campos ticket_id, subject, description, etc.
        questions: Preguntas tipadas opcionales. Si es None, usa las preguntas por defecto.

    Returns:
        Estructura formateada con state, questions y targets de evaluación.
    """
    active_questions = questions if questions is not None else build_default_laya_questions()

    subject_text = str(record.get("subject", "") or "").strip()
    body_text = str(record.get("description", "") or "").strip()

    # Si el subject está vacío, componemos un placeholder explícito para el estado de Laya
    formatted_state: Dict[str, str] = {
        "subject": subject_text if subject_text else "Sin asunto especificado",
        "body": body_text,
    }

    targets: Dict[str, str] = {
        "category": str(record.get("category", "unknown")),
        "priority": str(record.get("priority", "unknown")),
        "support_group": str(record.get("support_group", "unknown")),
    }

    return {
        "ticket_id": str(record.get("ticket_id", "")),
        "source_id": str(record.get("source_id", "")),
        "language": str(record.get("language", "und")),
        "content_hash": str(record.get("content_hash", "")),
        "state": formatted_state,
        "questions": active_questions,
        "targets": targets,
    }


def export_laya_formatted_dataset(
    input_path: Path,
    output_path: Path,
    filter_language: Optional[str] = None,
) -> int:
    """Convierte un archivo JSONL canónico a formato Laya Router y lo exporta.

    Args:
        input_path: Ruta al archivo JSONL de entrada (ej: deduplicated_tickets.jsonl).
        output_path: Ruta de destino para el archivo JSONL formateado.
        filter_language: Filtro ISO de idioma opcional (ej: 'es'). Si es None, incluye todos.

    Returns:
        Número total de registros exportados.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_count = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw_line = line.strip()
            if not raw_line:
                continue

            record = json.loads(raw_line)
            if filter_language is not None and record.get("language") != filter_language:
                continue

            laya_item = format_record_for_laya(record)
            fout.write(json.dumps(laya_item, ensure_ascii=False) + "\n")
            exported_count += 1

    return exported_count
