"""
Pruebas unitarias para el módulo de descarga e ingestión de datasets (src/ingestion/download.py).
"""

from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from src.ingestion.download import (
    calculate_sha256,
    download_file,
    record_manifest,
    safe_extract_archive,
)


def test_calculate_sha256_valid(tmp_path: Path):
    """Verifica el cómputo correcto y determinista del hash SHA-256."""
    import hashlib

    content = "Laya Service Desk Reproducibility Test"
    test_file = tmp_path / "sample.txt"
    test_file.write_text(content, encoding="utf-8")
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert calculate_sha256(test_file) == expected_hash


def test_calculate_sha256_missing_file():
    """Verifica que se lance FileNotFoundError ante un archivo inexistente."""
    with pytest.raises(FileNotFoundError):
        calculate_sha256("ruta/inexistente/archivo.txt")


def test_safe_extract_zip(tmp_path: Path):
    """Verifica la extracción controlada de un archivo ZIP válido."""
    zip_path = tmp_path / "valid.zip"
    extract_dir = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test1.csv", "id,text\n1,hola")
        zf.writestr("subfolder/test2.csv", "id,text\n2,mundo")

    extracted = safe_extract_archive(zip_path, extract_dir)
    assert len(extracted) == 2
    assert (extract_dir / "test1.csv").exists()
    assert (extract_dir / "subfolder" / "test2.csv").exists()


def test_safe_extract_path_traversal(tmp_path: Path):
    """Verifica que la extracción bloquee intentos de Zip Slip (Path Traversal)."""
    malicious_zip = tmp_path / "malicious.zip"
    extract_dir = tmp_path / "out"

    with zipfile.ZipFile(malicious_zip, "w") as zf:
        # Miembro con ruta relativa que escapa del directorio destino
        zf.writestr("../escaped.txt", "contenido no autorizado")

    with pytest.raises(ValueError, match="Intento de path traversal detectado"):
        safe_extract_archive(malicious_zip, extract_dir)


def test_record_manifest(tmp_path: Path):
    """Verifica la creación y actualización incremental del manifiesto inmutable."""
    manifest_path = tmp_path / "manifest.json"
    files_info = [
        {"filename": "X_train.csv", "size_bytes": 1024, "sha256": "abc123hash"}
    ]

    manifest = record_manifest(
        manifest_path=manifest_path,
        source_id="zenodo_tickets",
        files_info=files_info,
        source_url="https://zenodo.org/api/records/7648117",
        license_type="CC-BY-4.0",
    )

    assert manifest["schema_version"] == 1
    assert "zenodo_tickets" in manifest["sources"]
    assert manifest["sources"]["zenodo_tickets"]["license"] == "CC-BY-4.0"
    assert manifest_path.exists()

    # Re-lectura del archivo guardado
    saved_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_data["sources"]["zenodo_tickets"]["files"][0]["filename"] == "X_train.csv"


def test_download_file_no_overwrite(tmp_path: Path):
    """Verifica que download_file lance FileExistsError si overwrite=False y check_existing=False."""
    existing_file = tmp_path / "data.csv"
    existing_file.write_text("ya_existe", encoding="utf-8")

    with pytest.raises(FileExistsError, match="El archivo destino ya existe"):
        download_file("http://example.com/data.csv", existing_file, overwrite=False, check_existing=False)


def test_download_file_skip_if_exists(tmp_path: Path):
    """Verifica la idempotencia: si el archivo ya existe, no se conecta a la red."""
    existing_file = tmp_path / "cached.csv"
    existing_file.write_text("datos_en_cache", encoding="utf-8")

    # URL inalcanzable que fallaría inmediatamente si intentase conectar
    result = download_file("http://host-inalcanzable.local/data.csv", existing_file, check_existing=True)
    assert result == existing_file


def test_download_file_mocked_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifica una descarga simulada exitosa con comprobación de hash SHA-256."""
    import hashlib

    target_file = tmp_path / "downloaded.txt"
    content = b"Contenido de prueba para descarga segura"
    expected_hash = hashlib.sha256(content).hexdigest()

    class MockResponse:
        def __init__(self, data: bytes):
            self.stream = BytesIO(data)

        def read(self, chunk_size: int = -1) -> bytes:
            return self.stream.read(chunk_size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_urlopen(req, timeout=60):
        return MockResponse(content)

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    result = download_file(
        url="http://test.local/downloaded.txt",
        target_path=target_file,
        expected_sha256=expected_hash,
    )

    assert result.exists()
    assert result.read_bytes() == content


def test_download_file_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifica que una discrepancia de hash lance ValueError y limpie el archivo temporal."""
    target_file = tmp_path / "corrupt.txt"
    content = b"Contenido corrupto"

    class MockResponse:
        def __init__(self, data: bytes):
            self.stream = BytesIO(data)

        def read(self, chunk_size: int = -1) -> bytes:
            return self.stream.read(chunk_size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=60: MockResponse(content))

    with pytest.raises(ValueError, match="Fallo de integridad"):
        download_file(
            url="http://test.local/corrupt.txt",
            target_path=target_file,
            expected_sha256="hash_invalido_que_no_coincide",
        )

    # El archivo destino y los temporales no deben quedar en disco
    assert not target_file.exists()
