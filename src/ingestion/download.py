"""
Módulo de descarga, verificación de integridad y registro de manifiestos de datasets.

Fase: Ingestión de Datos (Fase 1 del Roadmap)
Garantías:
    - Descarga por streaming mediante librería estándar (sin dependencias externas pesadas).
    - Cómputo y verificación estricta de hash SHA-256 en vuelo.
    - Escritura atómica vía archivos temporales (.tmp) para evitar corrupción por descargas truncadas.
    - Descompresión segura de archivos empaquetados contra vulnerabilidades Path Traversal.
    - Generación y mantenimiento de manifiesto inmutable en data/raw/manifest.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tarfile
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import zipfile

# DECISIÓN DE ARQUITECTURA: Usar logging estándar configurado en el módulo para permitir
# trazabilidad sin forzar dependencias externas en el entorno de desarrollo local.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 64 * 1024  # 64 KB
USER_AGENT: str = "LayaServiceDesk-Ingestion/1.0 (+https://github.com/NandhaKishorM/laya)"


def calculate_sha256(file_path: Path | str) -> str:
    """
    Calcula el hash criptográfico SHA-256 de un archivo local de forma eficiente por bloques.

    Args:
        file_path: Ruta al archivo a procesar.

    Returns:
        Cadena hexadecimal de 64 caracteres correspondiente al hash SHA-256.

    Raises:
        FileNotFoundError: Si el archivo no existe en el sistema de archivos.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Archivo no encontrado para cómputo de hash: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(DEFAULT_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    target_path: Path | str,
    expected_sha256: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """
    Descarga un archivo remoto vía HTTP/HTTPS con validación de integridad y escritura atómica.

    Args:
        url: URL completa del recurso a descargar.
        target_path: Destino final del archivo en disco.
        expected_sha256: Hash SHA-256 esperado opcional para validación de integridad.
        overwrite: Si es True, permite sobreescribir el archivo destino si ya existe.
        timeout: Tiempo máximo de espera en segundos para la conexión HTTP.

    Returns:
        Path al archivo final descargado y verificado.

    Raises:
        FileExistsError: Si el archivo ya existe y overwrite es False.
        ValueError: Si el hash calculado no coincide con expected_sha256.
        urllib.error.URLError: Si ocurre un fallo de red o error de respuesta HTTP.
    """
    target = Path(target_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo destino ya existe: {target}. Use overwrite=True para reemplazarlo."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp_{os.getpid()}")

    logger.info("Iniciando descarga desde %s hacia %s", url, target)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as out_file:
            while chunk := response.read(DEFAULT_CHUNK_SIZE):
                hasher.update(chunk)
                out_file.write(chunk)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error("Error durante la descarga de %s: %s", url, exc)
        raise

    calculated_hash = hasher.hexdigest()
    if expected_sha256 and calculated_hash.lower() != expected_sha256.lower():
        tmp_path.unlink()
        raise ValueError(
            f"Fallo de integridad para {target.name}: esperado {expected_sha256}, obtenido {calculated_hash}"
        )

    # Reemplazo atómico seguro
    if target.exists():
        target.unlink()
    tmp_path.rename(target)
    logger.info("Descarga completada con éxito. SHA-256: %s", calculated_hash)
    return target


def safe_extract_archive(
    archive_path: Path | str,
    extract_dir: Path | str,
    remove_archive_after: bool = False,
) -> List[Path]:
    """
    Descomprime un archivo .zip o .tar.gz protegiendo contra vulnerabilidades de Path Traversal (Zip Slip).

    Args:
        archive_path: Ruta al archivo comprimido.
        extract_dir: Directorio de destino para los archivos extraídos.
        remove_archive_after: Si es True, elimina el archivo comprimido tras la descompresión.

    Returns:
        Lista de Paths de todos los archivos extraídos exitosamente.

    Raises:
        ValueError: Si el formato del archivo no está soportado o detecta un path inseguro.
    """
    archive = Path(archive_path)
    out_dir = Path(extract_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: List[Path] = []

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                target_dest = (out_dir / member.filename).resolve()
                if not str(target_dest).startswith(str(out_dir)):
                    raise ValueError(f"Intento de path traversal detectado en archivo ZIP: {member.filename}")
                zf.extract(member, out_dir)
                if not member.is_dir():
                    extracted_files.append(target_dest)

    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            for member in tf.getmembers():
                target_dest = (out_dir / member.name).resolve()
                if not str(target_dest).startswith(str(out_dir)):
                    raise ValueError(f"Intento de path traversal detectado en archivo TAR: {member.name}")
                tf.extract(member, out_dir)
                if member.isfile():
                    extracted_files.append(target_dest)
    else:
        raise ValueError(f"Formato de archivo no soportado para descompresión: {archive}")

    if remove_archive_after and archive.exists():
        archive.unlink()

    logger.info("Extraídos %d archivos en %s", len(extracted_files), out_dir)
    return extracted_files


def record_manifest(
    manifest_path: Path | str,
    source_id: str,
    files_info: List[Dict[str, Any]],
    source_url: str,
    license_type: str,
) -> Dict[str, Any]:
    """
    Registra de forma inmutable la metadata de los archivos descargados en el manifiesto JSON.

    Args:
        manifest_path: Ruta al archivo manifest.json.
        source_id: Identificador de la fuente (ej. zenodo_tickets).
        files_info: Lista de metadatos de los archivos (nombre, tamaño en bytes, hash sha256).
        source_url: URL de procedencia o repositorio.
        license_type: Licencia legal verificada de la fuente.

    Returns:
        Diccionario completo del manifiesto actualizado.
    """
    manifest_file = Path(manifest_path)
    manifest_data: Dict[str, Any] = {
        "schema_version": 1,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("No se pudo leer el manifiesto previo (%s). Se creará uno nuevo.", exc)

    manifest_data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_data.setdefault("sources", {})[source_id] = {
        "source_id": source_id,
        "source_url": source_url,
        "license": license_type,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
    }

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica de manifiesto
    tmp_manifest = manifest_file.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    if manifest_file.exists():
        manifest_file.unlink()
    tmp_manifest.rename(manifest_file)

    logger.info("Manifiesto actualizado para la fuente '%s' en %s", source_id, manifest_file)
    return manifest_data


def download_zenodo_dataset(
    record_id: str,
    output_dir: Path | str,
    allowed_files: Optional[List[str]] = None,
    overwrite: bool = False,
    manifest_path: Optional[Path | str] = None,
) -> List[Path]:
    """
    Descarga archivos CSV o datos desde la API pública de Zenodo para un record específico.

    Args:
        record_id: Identificador numérico del record en Zenodo (ej. '7648117').
        output_dir: Directorio donde almacenar los archivos descargados.
        allowed_files: Lista de nombres de archivo permitidos para filtrar descargas accesorias.
        overwrite: Si es True, permite sobreescribir archivos existentes.
        manifest_path: Ruta opcional para actualizar el manifiesto inmutable.

    Returns:
        Lista de Paths correspondientes a los archivos descargados y verificados.

    Raises:
        ValueError: Si el registro no contiene archivos o no se puede consultar la API.
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logger.info("Consultando metadata de Zenodo en %s", api_url)

    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        record_metadata = json.loads(resp.read().decode("utf-8"))

    files_list = record_metadata.get("files", [])
    if not files_list:
        raise ValueError(f"El registro de Zenodo {record_id} no contiene archivos descargables.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths: List[Path] = []
    manifest_files_info: List[Dict[str, Any]] = []

    for file_meta in files_list:
        file_key = file_meta.get("key", "")
        if allowed_files and file_key not in allowed_files:
            continue

        download_link = file_meta.get("links", {}).get("self")
        if not download_link:
            continue

        target_file = target_dir / file_key
        # En Zenodo el link suele ser /content
        content_url = download_link if download_link.endswith("/content") else f"{download_link}/content"

        saved_path = download_file(content_url, target_file, overwrite=overwrite)
        downloaded_paths.append(saved_path)

        sha256_hash = calculate_sha256(saved_path)
        manifest_files_info.append(
            {
                "filename": file_key,
                "size_bytes": saved_path.stat().st_size,
                "sha256": sha256_hash,
                "download_url": content_url,
            }
        )

    if manifest_path and manifest_files_info:
        record_manifest(
            manifest_path=manifest_path,
            source_id="zenodo_tickets",
            files_info=manifest_files_info,
            source_url=api_url,
            license_type="CC-BY-4.0",
        )

    return downloaded_paths


def main() -> None:
    """Punto de entrada CLI para la ejecución interactiva o por script de descargas."""
    parser = argparse.ArgumentParser(description="Módulo de descarga de datasets para Laya Service Desk.")
    parser.add_argument("--source", choices=["zenodo", "generic"], default="zenodo", help="Fuente a descargar.")
    parser.add_argument("--record-id", default="7648117", help="ID de Zenodo si aplica.")
    parser.add_argument("--output-dir", default="data/raw/zenodo_tickets", help="Directorio destino.")
    parser.add_argument("--manifest", default="data/raw/manifest.json", help="Ruta del manifiesto.")
    parser.add_argument("--overwrite", action="store_true", help="Permitir sobreescritura de archivos existentes.")
    args = parser.parse_args()

    if args.source == "zenodo":
        download_zenodo_dataset(
            record_id=args.record_id,
            output_dir=args.output_dir,
            allowed_files=["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"],
            overwrite=args.overwrite,
            manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()
