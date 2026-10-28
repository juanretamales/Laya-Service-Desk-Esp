"""Tests unitarios para el particionado estratificado y verificación de zero leakage."""

from pathlib import Path
import pytest

from src.preprocessing.split import (
    calculate_file_sha256,
    export_splits,
    stratified_split,
    verify_zero_leakage,
)


def test_stratified_split_proportions():
    """Valida que las proporciones y tamaños de los splits sean consistentes."""
    records = [
        {"ticket_id": f"t_{i}", "category": "software" if i < 60 else "hardware", "content_hash": f"hash_{i}"}
        for i in range(100)
    ]

    train, val, test = stratified_split(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) + len(val) + len(test) == 100
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Validar que ambas clases estén representadas en cada split
    train_cats = {r["category"] for r in train}
    val_cats = {r["category"] for r in val}
    test_cats = {r["category"] for r in test}

    assert train_cats == {"software", "hardware"}
    assert val_cats == {"software", "hardware"}
    assert test_cats == {"software", "hardware"}


def test_stratified_split_invalid_ratios():
    """Valida que falle si las proporciones no suman 1.0."""
    records = [{"ticket_id": "1", "category": "software", "content_hash": "h1"}]
    with pytest.raises(ValueError, match="Las proporciones deben sumar 1.0"):
        stratified_split(records, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_stratified_split_empty_records():
    """Valida que falle si se le pasa una lista vacía."""
    with pytest.raises(ValueError, match="No se pueden particionar registros de una lista vacía"):
        stratified_split([])


def test_verify_zero_leakage_success():
    """Valida la verificación exitosa cuando los hashes son disjuntos."""
    train = [{"content_hash": "h1"}, {"content_hash": "h2"}]
    val = [{"content_hash": "h3"}]
    test = [{"content_hash": "h4"}, {"content_hash": "h5"}]

    assert verify_zero_leakage(train, val, test) is True


def test_verify_zero_leakage_detected():
    """Valida que lance error si existe fuga entre particiones."""
    train = [{"content_hash": "h1"}, {"content_hash": "h_leak"}]
    val = [{"content_hash": "h3"}]
    test = [{"content_hash": "h4"}, {"content_hash": "h_leak"}]

    with pytest.raises(ValueError, match="Fuga de datos detectada entre Train y Test"):
        verify_zero_leakage(train, val, test)


def test_export_splits_and_manifest(tmp_path: Path):
    """Valida la exportación de splits, generación de hashes SHA-256 y manifiesto."""
    train = [
        {"ticket_id": "t1", "category": "software", "content_hash": "h1"},
        {"ticket_id": "t2", "category": "access", "content_hash": "h2"},
    ]
    val = [{"ticket_id": "t3", "category": "software", "content_hash": "h3"}]
    test = [{"ticket_id": "t4", "category": "access", "content_hash": "h4"}]

    out_dir = tmp_path / "data" / "splits"
    manifest = export_splits(train, val, test, out_dir)

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (out_dir / "splits_manifest.json").exists()

    assert manifest["total_records"] == 4
    assert manifest["splits"]["train"]["count"] == 2
    assert manifest["splits"]["val"]["count"] == 1
    assert manifest["splits"]["test"]["count"] == 1

    # Validar que los hashes en el manifiesto coincidan con los archivos reales
    for split_key in ("train", "val", "test"):
        sha_recorded = manifest["splits"][split_key]["sha256"]
        real_sha = calculate_file_sha256(out_dir / f"{split_key}.jsonl")
        assert sha_recorded == real_sha
