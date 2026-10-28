"""
Módulo de deduplicación determinista de tickets para Laya Service Desk.

Fase: Preprocesamiento y Calidad de Datos (Fase 2.4 del Roadmap)
Garantías:
    - Deduplicación exacta basada en hash SHA-256 de texto normalizado (asunto + descripción).
    - Normalización de caracteres y puntuación para capturar duplicados con variaciones triviales.
    - Preservación del primer registro canónico y aislamiento de duplicados para auditoría.
    - Generación de estadísticas y métricas de duplicidad por fuente de datos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# DECISIÓN DE ARQUITECTURA: Normalizar espacios, caracteres de escape y puntuación periférica
# antes de calcular el hash criptográfico para evitar que tickets idénticos con saltos de línea
# o puntuación menor evadan la deduplicación.
PUNCTUATION_CLEANER: re.Pattern[str] = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text_for_hashing(text: str) -> str:
    """
    Normaliza una cadena textual para cálculo de similitud y deduplicación.

    Args:
        text: Cadena textual cruda.

    Returns:
        Texto en minúsculas, sin puntuación periférica y con espacios colapsados.
    """
    if not text:
        return ""
    cleaned = PUNCTUATION_CLEANER.sub(" ", str(text).lower())
    return " ".join(cleaned.split())


def compute_content_hash(
    record: Dict[str, Any],
    text_fields: Sequence[str] = ("subject", "description"),
) -> str:
    """
    Calcula el hash criptográfico SHA-256 del contenido combinado de los campos textuales de un ticket.

    Args:
        record: Diccionario representativo del ticket.
        text_fields: Secuencia de claves textuales a incluir en la huella.

    Returns:
        Cadena hexadecimal de 64 caracteres correspondiente al hash SHA-256 del contenido.
    """
    parts = [normalize_text_for_hashing(str(record.get(field, "") or "")) for field in text_fields]
    combined = " || ".join(p for p in parts if p)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def deduplicate_records(
    records: List[Dict[str, Any]],
    text_fields: Sequence[str] = ("subject", "description"),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplica una lista de registros conservando la primera aparición y aislando los duplicados.

    Args:
        records: Lista de diccionarios de tickets a procesar.
        text_fields: Claves textuales utilizadas para generar la huella de unicidad.

    Returns:
        Tupla con:
            - unique_records: Lista de tickets únicos.
            - duplicate_records: Lista de tickets duplicados descartados con referencia al original.
            - stats: Diccionario con estadísticas agregadas de duplicidad por fuente.
    """
    seen_hashes: Dict[str, str] = {}  # hash -> ticket_id del primer registro visto
    unique_records: List[Dict[str, Any]] = []
    duplicate_records: List[Dict[str, Any]] = []

    duplicates_by_source: Dict[str, int] = {}
    total_by_source: Dict[str, int] = {}

    for rec in records:
        source_id = str(rec.get("source_id", "unknown"))
        ticket_id = str(rec.get("ticket_id", "unknown"))
        total_by_source[source_id] = total_by_source.get(source_id, 0) + 1

        content_hash = compute_content_hash(rec, text_fields)

        if content_hash in seen_hashes:
            dup_info = dict(rec)
            dup_info["duplicate_of_ticket_id"] = seen_hashes[content_hash]
            dup_info["content_hash"] = content_hash
            duplicate_records.append(dup_info)
            duplicates_by_source[source_id] = duplicates_by_source.get(source_id, 0) + 1
        else:
            seen_hashes[content_hash] = ticket_id
            enriched_rec = dict(rec)
            enriched_rec["content_hash"] = content_hash
            unique_records.append(enriched_rec)

    total_records = len(records)
    unique_count = len(unique_records)
    duplicate_count = len(duplicate_records)
    ratio = round(duplicate_count / total_records, 4) if total_records > 0 else 0.0

    stats: Dict[str, Any] = {
        "total_records": total_records,
        "unique_records": unique_count,
        "duplicate_records": duplicate_count,
        "duplicate_ratio": ratio,
        "duplicates_by_source": duplicates_by_source,
        "total_by_source": total_by_source,
    }

    logger.info(
        "Deduplicación finalizada: %d totales -> %d únicos, %d duplicados (%.2f%%)",
        total_records,
        unique_count,
        duplicate_count,
        ratio * 100,
    )
    return unique_records, duplicate_records, stats


def export_deduplication_results(
    unique_records: List[Dict[str, Any]],
    duplicate_records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_clean_jsonl: Path | str,
    output_duplicates_jsonl: Optional[Path | str] = None,
    output_stats_json: Optional[Path | str] = None,
    overwrite: bool = False,
) -> None:
    """
    Persiste en disco los registros deduplicados, los duplicados auditados y el resumen de estadísticas.

    Args:
        unique_records: Lista de registros únicos a guardar.
        duplicate_records: Lista de registros duplicados.
        stats: Diccionario de estadísticas.
        output_clean_jsonl: Ruta destino para los registros limpios.
        output_duplicates_jsonl: Ruta opcional para guardar registros duplicados.
        output_stats_json: Ruta opcional para guardar el reporte estadístico.
        overwrite: Si es True, permite sobreescribir archivos existentes.

    Raises:
        FileExistsError: Si alguno de los archivos destino ya existe y overwrite es False.
    """
    clean_path = Path(output_clean_jsonl)
    if clean_path.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo destino ya existe: {clean_path}. Use overwrite=True para reemplazarlo."
        )

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_clean = clean_path.with_suffix(".tmp")
    with tmp_clean.open("w", encoding="utf-8") as f:
        for rec in unique_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if clean_path.exists():
        clean_path.unlink()
    tmp_clean.rename(clean_path)

    if output_duplicates_jsonl and duplicate_records:
        dup_path = Path(output_duplicates_jsonl)
        dup_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dup = dup_path.with_suffix(".tmp")
        with tmp_dup.open("w", encoding="utf-8") as f:
            for rec in duplicate_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if dup_path.exists():
            dup_path.unlink()
        tmp_dup.rename(dup_path)

    if output_stats_json:
        stats_path = Path(output_stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Punto de entrada CLI para la deduplicación de datasets procesados."""
    parser = argparse.ArgumentParser(description="Deduplicador determinista de tickets para Laya.")
    parser.add_argument("--input", default="data/processed/unified_tickets.jsonl", help="Ruta del JSONL de entrada.")
    parser.add_argument("--output", default="data/processed/deduplicated_tickets.jsonl", help="Ruta de salida única.")
    parser.add_argument("--duplicates-output", default="data/processed/duplicate_tickets.jsonl", help="Ruta dups.")
    parser.add_argument("--stats-output", default="data/processed/deduplication_stats.json", help="Ruta reporte stats.")
    parser.add_argument("--overwrite", action="store_true", help="Permitir sobreescritura de archivos destino.")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_file}")

    records: List[Dict[str, Any]] = [
        json.loads(line) for line in input_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()
    ]

    unique_recs, dup_recs, stats = deduplicate_records(records)
    export_deduplication_results(
        unique_records=unique_recs,
        duplicate_records=dup_recs,
        stats=stats,
        output_clean_jsonl=args.output,
        output_duplicates_jsonl=args.duplicates_output,
        output_stats_json=args.stats_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
