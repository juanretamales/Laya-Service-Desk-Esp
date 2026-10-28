"""
Módulo de unificación de datasets, estandarización de esquemas y mapeo semántico de etiquetas.

Fase: Preprocesamiento y Calidad de Datos (Fase 2.3 del Roadmap)
Garantías:
    - Estandarización de columnas hacia el contrato canónico:
      (ticket_id, subject, description, category, priority, support_group).
    - Mapeo semántico de etiquetas hacia el vocabulario controlado de docs/04_NORMALIZACION.md.
    - Asignación explícita de 'unknown' ante etiquetas faltantes o no inferibles (prohibido inventar clases).
    - Integración determinista con anonimización PII y detección de idioma.
    - Prevención de sobreescritura accidental en data/processed/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.preprocessing.anonymize import anonymize_text
from src.preprocessing.language import detect_language, is_spanish

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# DECISIÓN DE ARQUITECTURA: Vocabularios canónicos inmutables alineados con docs/04_NORMALIZACION.md.
# Cualquier etiqueta fuera de estos conjuntos debe mapearse estrictamente a 'unknown'.
CANONICAL_CATEGORIES: Tuple[str, ...] = (
    "access",
    "network",
    "hardware",
    "software",
    "email",
    "database",
    "other",
    "unknown",
)

CANONICAL_PRIORITIES: Tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "critical",
    "unknown",
)

CANONICAL_SUPPORT_GROUPS: Tuple[str, ...] = (
    "service_desk",
    "identity",
    "network",
    "workplace",
    "applications",
    "dba",
    "other",
    "unknown",
)

REQUIRED_FIELDS: Tuple[str, ...] = (
    "ticket_id",
    "subject",
    "description",
    "category",
    "priority",
    "support_group",
)

# Diccionario semántico para mapear etiquetas de Zenodo (Record 7648117)
ZENODO_CATEGORY_MAP: Dict[str, str] = {
    "active directory": "access",
    "fileservice": "software",
    "software": "software",
    "o365": "email",
    "computer-services": "hardware",
    "eol": "hardware",
    "support general": "other",
}

# Diccionario semántico para prioridades genéricas
PRIORITY_MAP: Dict[str, str] = {
    "1": "low",
    "low": "low",
    "baja": "low",
    "minor": "low",
    "2": "medium",
    "medium": "medium",
    "media": "medium",
    "normal": "medium",
    "3": "high",
    "high": "high",
    "alta": "high",
    "major": "high",
    "4": "critical",
    "critical": "critical",
    "critica": "critical",
    "crítica": "critical",
    "urgent": "critical",
    "urgente": "critical",
}

# Diccionario semántico para grupos de soporte genéricos
SUPPORT_GROUP_MAP: Dict[str, str] = {
    "servicedesk": "service_desk",
    "service desk": "service_desk",
    "mesa de ayuda": "service_desk",
    "support general": "service_desk",
    "identity": "identity",
    "identidad": "identity",
    "active directory": "identity",
    "seguridad": "identity",
    "network": "network",
    "redes": "network",
    "telecom": "network",
    "workplace": "workplace",
    "hardware": "workplace",
    "puesto de trabajo": "workplace",
    "computer-services": "workplace",
    "applications": "applications",
    "software": "applications",
    "aplicaciones": "applications",
    "erp": "applications",
    "dba": "dba",
    "database": "dba",
    "bases de datos": "dba",
}


def map_canonical_label(raw_value: Optional[str], mapping: Dict[str, str], allowed_set: Sequence[str]) -> str:
    """
    Normaliza y mapea un valor crudo a su clase canónica correspondiente.

    Args:
        raw_value: Cadena original de la etiqueta en la fuente.
        mapping: Diccionario de correspondencia semántica.
        allowed_set: Conjunto de etiquetas canónicas válidas.

    Returns:
        Etiqueta canónica normalizada o 'unknown' si no coincide.
    """
    if not raw_value:
        return "unknown"

    cleaned = str(raw_value).strip().lower()
    if cleaned in mapping:
        return mapping[cleaned]

    if cleaned in allowed_set:
        return cleaned

    return "unknown"


def build_unified_record(
    ticket_id: str,
    subject: str,
    description: str,
    category: str,
    priority: str,
    support_group: str,
    source_id: str,
    anonymize: bool = True,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construye y valida un registro unificado conforme al contrato del laboratorio.

    Args:
        ticket_id: Identificador único del ticket en la fuente.
        subject: Asunto o título breve.
        description: Cuerpo o descripción detallada del problema.
        category: Etiqueta canónica de categoría.
        priority: Etiqueta canónica de prioridad.
        support_group: Etiqueta canónica de grupo resolutor.
        source_id: Identificador de la fuente para trazabilidad.
        anonymize: Si es True, aplica sanitización PII sobre asunto y descripción.
        language_hint: Pista de idioma provista por la fuente si existe.

    Returns:
        Diccionario con el registro completamente estandarizado y enriquecido.

    Raises:
        ValueError: Si el ticket carece de ID o no contiene texto descriptivo alguno.
    """
    tid = str(ticket_id or "").strip()
    subj = str(subject or "").strip()
    desc = str(description or "").strip()

    if not tid:
        raise ValueError(f"[{source_id}] Registro inválido: ticket_id vacío.")
    if not subj and not desc:
        raise ValueError(f"[{source_id}] Registro inválido: ticket {tid} sin asunto ni descripción.")

    if anonymize:
        subj = anonymize_text(subj)
        desc = anonymize_text(desc)

    combined_text = f"{subj} {desc}".strip()
    detected_lang, lang_conf = detect_language(combined_text)
    effective_lang = language_hint if language_hint else detected_lang

    cat = category if category in CANONICAL_CATEGORIES else "unknown"
    prio = priority if priority in CANONICAL_PRIORITIES else "unknown"
    group = support_group if support_group in CANONICAL_SUPPORT_GROUPS else "unknown"

    return {
        "ticket_id": tid,
        "subject": subj,
        "description": desc,
        "category": cat,
        "priority": prio,
        "support_group": group,
        "source_id": source_id,
        "language": effective_lang,
        "language_confidence": lang_conf,
    }


def load_zenodo_records(
    zenodo_dir: Path | str,
    anonymize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Carga, cruza y unifica las particiones crudas de Zenodo (X_train/y_train y X_test/y_test).

    Args:
        zenodo_dir: Directorio que contiene los archivos CSV de Zenodo.
        anonymize: Si es True, asegura la sanitización PII de los textos.

    Returns:
        Lista de registros estandarizados listos para consolidación.
    """
    base_path = Path(zenodo_dir)
    records: List[Dict[str, Any]] = []

    pairs = [
        ("X_train.csv", "y_train.csv", "zenodo_train"),
        ("X_test.csv", "y_test.csv", "zenodo_test"),
    ]

    for x_file, y_file, split_tag in pairs:
        x_path = base_path / x_file
        y_path = base_path / y_file

        if not x_path.exists() or not y_path.exists():
            logger.warning("No se encontraron ambos archivos de Zenodo: %s y %s", x_path, y_path)
            continue

        # 1. Cargar etiquetas
        labels_by_id: Dict[str, str] = {}
        with y_path.open("r", encoding="utf-8-sig") as f_y:
            reader = csv.DictReader(f_y)
            for row in reader:
                labels_by_id[row["id"].strip()] = row.get("category_truth", "")

        # 2. Cargar features y cruzar por ID
        with x_path.open("r", encoding="utf-8-sig") as f_x:
            reader = csv.DictReader(f_x)
            for row in reader:
                tid = row["id"].strip()
                raw_text = row.get("text", "")
                raw_cat = labels_by_id.get(tid, "")

                canonical_cat = map_canonical_label(raw_cat, ZENODO_CATEGORY_MAP, CANONICAL_CATEGORIES)

                record = build_unified_record(
                    ticket_id=f"zenodo_{tid}",
                    subject="",  # Zenodo no tiene campo subject separado; todo reside en text
                    description=raw_text,
                    category=canonical_cat,
                    priority="unknown",  # Zenodo no etiqueta prioridad
                    support_group="unknown",  # Zenodo no etiqueta grupo resolutor
                    source_id="zenodo_tickets",
                    anonymize=anonymize,
                )
                records.append(record)

    logger.info("Cargados %d registros de Zenodo desde %s", len(records), base_path)
    return records


def export_unified_jsonl(
    records: List[Dict[str, Any]],
    output_path: Path | str,
    overwrite: bool = False,
) -> Path:
    """
    Exporta una colección de registros unificados en formato JSONL delimitado por saltos de línea.

    Args:
        records: Lista de diccionarios de tickets normalizados.
        output_path: Ruta destino del archivo .jsonl.
        overwrite: Si es True, permite reemplazar un archivo existente.

    Returns:
        Path al archivo exportado.

    Raises:
        FileExistsError: Si el archivo ya existe y overwrite=False.
    """
    out_file = Path(output_path)
    if out_file.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo unificado ya existe: {out_file}. Pase overwrite=True para reemplazarlo."
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_file.with_suffix(".tmp")

    with tmp_out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_file.exists():
        out_file.unlink()
    tmp_out.rename(out_file)

    logger.info("Exportados exitosamente %d registros hacia %s", len(records), out_file)
    return out_file


def main() -> None:
    """Punto de entrada de línea de comandos para la unificación de datasets."""
    parser = argparse.ArgumentParser(description="Unificación y mapeo semántico de tickets para Laya.")
    parser.add_argument("--zenodo-dir", default="data/raw/zenodo_tickets", help="Directorio con CSVs de Zenodo.")
    parser.add_argument("--output", default="data/processed/unified_tickets.jsonl", help="Ruta de salida JSONL.")
    parser.add_argument("--filter-spanish", action="store_true", help="Filtrar solo registros en español.")
    parser.add_argument("--overwrite", action="store_true", help="Permitir sobreescritura de salida.")
    args = parser.parse_args()

    records = load_zenodo_records(args.zenodo_dir, anonymize=True)

    if args.filter_spanish:
        initial_count = len(records)
        records = [r for r in records if is_spanish(f"{r['subject']} {r['description']}")]
        logger.info("Filtro de español aplicado: %d -> %d registros conservados.", initial_count, len(records))

    export_unified_jsonl(records, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
