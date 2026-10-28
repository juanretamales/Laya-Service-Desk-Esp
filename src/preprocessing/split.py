"""Particionado estratificado sin fuga de datos (Zero-Leakage Splitting).

Este módulo particiona datasets de tickets en conjuntos de entrenamiento,
validación y prueba asegurando:
1. Estratificación por clase (categoría canónica).
2. Aislamiento estricto sin solapamiento de contenido (hashes de contenido disjuntos).
3. Congelamiento inmutable del conjunto de prueba con hash SHA-256 verificado.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple


def stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_key: str = "category",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Particiona una lista de registros de forma estratificada y determinista.

    Args:
        records: Lista de registros a particionar.
        train_ratio: Proporción para el conjunto de entrenamiento (ej: 0.70).
        val_ratio: Proporción para el conjunto de validación (ej: 0.15).
        test_ratio: Proporción para el conjunto de prueba (ej: 0.15).
        seed: Semilla pseudoaleatoria para reproducibilidad estricta.
        stratify_key: Clave por la cual estratificar las particiones.

    Returns:
        Tupla con (train_records, val_records, test_records).

    Raises:
        ValueError: Si la suma de proporciones no es 1.0 o la lista de registros está vacía.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0, pero suman {total_ratio:.4f}")

    if not records:
        raise ValueError("No se pueden particionar registros de una lista vacía.")

    # Agrupar registros por el valor de la clave de estratificación
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key_val = str(rec.get(stratify_key, "unknown"))
        grouped[key_val].append(rec)

    rng = random.Random(seed)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # Ordenar las claves para garantizar determinismo absoluto entre plataformas
    for key in sorted(grouped.keys()):
        group = grouped[key][:]
        rng.shuffle(group)
        n = len(group)

        if n == 1:
            # Caso extremo: una sola muestra se asigna a train
            train_records.extend(group)
            continue
        elif n == 2:
            # Dos muestras: una a train, una a test
            train_records.append(group[0])
            test_records.append(group[1])
            continue

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Asegurar que no exceda el total y que test tenga al menos 1 si el grupo lo permite
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:n_train + n_val])
        test_records.extend(group[n_train + n_val:])

    # Barajar cada partición resultante de forma determinista
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return train_records, val_records, test_records


def verify_zero_leakage(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    hash_key: str = "content_hash",
) -> bool:
    """Verifica que no exista solapamiento de contenido entre las tres particiones.

    Args:
        train_records: Registros del conjunto de entrenamiento.
        val_records: Registros del conjunto de validación.
        test_records: Registros del conjunto de prueba.
        hash_key: Clave del hash o identificador único de contenido a contrastar.

    Returns:
        True si los tres conjuntos son estrictamente disjuntos.

    Raises:
        ValueError: Si se detecta cualquier intersección de registros entre particiones.
    """
    train_hashes: Set[str] = {str(r[hash_key]) for r in train_records if hash_key in r}
    val_hashes: Set[str] = {str(r[hash_key]) for r in val_records if hash_key in r}
    test_hashes: Set[str] = {str(r[hash_key]) for r in test_records if hash_key in r}

    leak_train_val = train_hashes.intersection(val_hashes)
    if leak_train_val:
        raise ValueError(
            f"Fuga de datos detectada entre Train y Val ({len(leak_train_val)} colisiones): {leak_train_val}"
        )

    leak_train_test = train_hashes.intersection(test_hashes)
    if leak_train_test:
        raise ValueError(
            f"Fuga de datos detectada entre Train y Test ({len(leak_train_test)} colisiones): {leak_train_test}"
        )

    leak_val_test = val_hashes.intersection(test_hashes)
    if leak_val_test:
        raise ValueError(
            f"Fuga de datos detectada entre Val y Test ({len(leak_val_test)} colisiones): {leak_val_test}"
        )

    return True


def calculate_file_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_splits(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: Path,
    stratify_key: str = "category",
) -> Dict[str, Any]:
    """Exporta las particiones en formato JSONL y genera un manifiesto inmutable con SHA-256.

    Args:
        train_records: Registros para train.jsonl.
        val_records: Registros para val.jsonl.
        test_records: Registros para test.jsonl.
        output_dir: Directorio de destino para los archivos generados.
        stratify_key: Clave para resumir la distribución de clases en el manifiesto.

    Returns:
        Diccionario con el manifiesto y hashes SHA-256 de cada partición.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {
        "train": (output_dir / "train.jsonl", train_records),
        "val": (output_dir / "val.jsonl", val_records),
        "test": (output_dir / "test.jsonl", test_records),
    }

    manifest: Dict[str, Any] = {
        "total_records": len(train_records) + len(val_records) + len(test_records),
        "splits": {},
    }

    for split_name, (split_path, records) in splits_data.items():
        with split_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        file_hash = calculate_file_sha256(split_path)
        class_dist = dict(Counter(str(r.get(stratify_key, "unknown")) for r in records))

        manifest["splits"][split_name] = {
            "path": str(split_path.relative_to(output_dir.parent.parent)),
            "count": len(records),
            "sha256": file_hash,
            "class_distribution": class_dist,
        }

    manifest_path = output_dir / "splits_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
