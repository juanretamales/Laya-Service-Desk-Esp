"""Tests unitarios para el módulo de entrenamiento y fine-tuning de Laya."""

from pathlib import Path
import pytest
import yaml

from src.training.train import (
    load_jsonl_records,
    load_training_config,
    set_seed,
    train_laya,
)


def test_load_training_config_valid(tmp_path: Path):
    """Valida la carga correcta de un archivo de configuración válido."""
    cfg_file = tmp_path / "valid_training.yaml"
    cfg_data = {
        "model": {"family": "laya"},
        "data": {
            "train_file": "data/splits/train.jsonl",
            "val_file": "data/splits/val.jsonl",
        },
        "training": {"epochs": 3, "batch_size": 8},
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    loaded = load_training_config(cfg_file)
    assert loaded["training"]["epochs"] == 3
    assert loaded["data"]["train_file"] == "data/splits/train.jsonl"


def test_load_training_config_missing_file():
    """Valida que lance FileNotFoundError si no existe el archivo."""
    with pytest.raises(FileNotFoundError):
        load_training_config(Path("non_existent_config.yaml"))


def test_load_training_config_test_leakage_rejection(tmp_path: Path):
    """Valida que rechace configuraciones que intenten usar el test set para entrenar."""
    cfg_file = tmp_path / "leak_training.yaml"
    cfg_data = {
        "data": {
            "train_file": "data/splits/test.jsonl",
            "val_file": "data/splits/val.jsonl",
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Violación de protocolo de aislamiento"):
        load_training_config(cfg_file)


def test_set_seed_reproducibility():
    """Valida que set_seed garantice determinismo."""
    import random
    set_seed(123)
    val1 = random.random()
    set_seed(123)
    val2 = random.random()
    assert val1 == val2


def test_load_jsonl_records(tmp_path: Path):
    """Valida la lectura estructurada de registros JSONL."""
    jsonl_file = tmp_path / "samples.jsonl"
    jsonl_file.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    recs = load_jsonl_records(jsonl_file)
    assert len(recs) == 2
    assert recs[0]["id"] == 1


def test_train_laya_dry_run(tmp_path: Path):
    """Valida la ejecución de un paso dry-run de entrenamiento con artefactos exportados."""
    # Crear minidatasets de prueba
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    ckpt_dir = tmp_path / "checkpoints"

    import json
    sample_ticket = {
        "ticket_id": "test_1",
        "content_hash": "hash_t1",
        "state": {"subject": "Fallo Outlook", "body": "No envía emails"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Categoría",
                "criteria": ["email", "hardware", "software", "unknown"],
            }
        },
        "targets": {"category": "email"},
    }

    sample_val_ticket = {
        "ticket_id": "val_1",
        "content_hash": "hash_v1",
        "state": {"subject": "Pantalla rota", "body": "Monitor parpadea"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Categoría",
                "criteria": ["email", "hardware", "software", "unknown"],
            }
        },
        "targets": {"category": "hardware"},
    }

    with train_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_ticket) + "\n")
    with val_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_val_ticket) + "\n")

    cfg_file = tmp_path / "test_cfg.yaml"
    cfg_data = {
        "data": {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "tasks": ["category"],
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate_encoder": 1e-5,
            "learning_rate_head": 1e-4,
            "weight_decay": 0.01,
        },
        "output": {
            "checkpoint_dir": str(ckpt_dir),
        },
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    report = train_laya(config_path=cfg_file, device_name="cpu", dry_run=True, override_epochs=1)

    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "best_model.safetensors").exists()
    assert (ckpt_dir / "last_model.safetensors").exists()
    assert (ckpt_dir / "rl_agent_config.json").exists()
    assert (ckpt_dir / "training_report.json").exists()
    assert (ckpt_dir / "training_curves.json").exists()
    assert (ckpt_dir / "training_curves.csv").exists()
    assert (ckpt_dir / "figures" / "training_loss_curve.png").exists()
    assert (ckpt_dir / "figures" / "training_metrics_curve.png").exists()
    assert (ckpt_dir / "figures" / "training_dashboard.png").exists()
    assert "checkpoints" in report
    assert "early_stopping" in report
    assert "data_hashes" in report
    assert "visualizations" in report


def test_train_laya_early_stopping(tmp_path: Path):
    """Valida la detención temprana (Early Stopping) tras épocas sin mejora."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    ckpt_dir = tmp_path / "checkpoints_es"

    import json
    sample_train = {
        "ticket_id": "t1",
        "content_hash": "h_train_1",
        "state": {"subject": "Problema", "body": "Detalle"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Cat",
                "criteria": ["email", "hardware", "software", "unknown"],
            }
        },
        "targets": {"category": "software"},
    }

    sample_val = {
        "ticket_id": "v1",
        "content_hash": "h_val_1",
        "state": {"subject": "Pantalla", "body": "No enciende"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Cat",
                "criteria": ["email", "hardware", "software", "unknown"],
            }
        },
        "targets": {"category": "hardware"},
    }

    with train_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_train) + "\n")
    with val_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_val) + "\n")

    cfg_file = tmp_path / "cfg_es.yaml"
    cfg_data = {
        "data": {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "tasks": ["category"],
        },
        "training": {
            "seed": 42,
            "epochs": 5,
            "batch_size": 1,
            "early_stopping_patience": 1,
            "learning_rate_encoder": 0.0,
            "learning_rate_head": 0.0,
            "weight_decay": 0.0,
        },
        "output": {
            "checkpoint_dir": str(ckpt_dir),
        },
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    # Ejecutar con paciencia 1 y 5 épocas planeadas
    report = train_laya(
        config_path=cfg_file,
        device_name="cpu",
        dry_run=True,
        override_epochs=5,
        override_patience=1,
    )

    # Con paciencia 1, la época 1 establece el mejor resultado (1.0). La época 2 no mejora (empata o no supera),
    # por lo que debe activar early stopping y ejecutar como máximo 2 épocas de las 5 planeadas.
    assert report["early_stopping"]["stopped_early"] is True
    assert (ckpt_dir / "best_model.safetensors").exists()
    assert (ckpt_dir / "last_model.safetensors").exists()


def test_train_laya_resume(tmp_path: Path):
    """Valida la capacidad de reanudar entrenamiento (resume) usando last_model y training_curves.csv."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    ckpt_dir = tmp_path / "checkpoints_resume"

    import json
    sample_train = {
        "ticket_id": "tr_1",
        "content_hash": "hash_tr_1",
        "state": {"subject": "Red caída", "body": "No hay conexión"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Cat",
                "criteria": ["network", "software", "hardware", "unknown"],
            }
        },
        "targets": {"category": "network"},
    }
    sample_val = {
        "ticket_id": "va_1",
        "content_hash": "hash_va_1",
        "state": {"subject": "Router caído", "body": "Sin señal"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Cat",
                "criteria": ["network", "software", "hardware", "unknown"],
            }
        },
        "targets": {"category": "network"},
    }

    with train_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_train) + "\n")
    with val_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_val) + "\n")

    cfg_file = tmp_path / "cfg_resume.yaml"
    cfg_data = {
        "data": {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "tasks": ["category"],
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate_encoder": 1e-5,
            "learning_rate_head": 1e-4,
        },
        "output": {
            "checkpoint_dir": str(ckpt_dir),
        },
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    # Sesión 1: Ejecutar 1 época inicial
    rep1 = train_laya(cfg_file, device_name="cpu", dry_run=True, override_epochs=1, override_resume=False)
    assert rep1["hyperparameters"]["start_epoch"] == 1
    assert rep1["hyperparameters"]["end_epoch"] == 1
    assert rep1["hyperparameters"]["total_epochs_in_history"] == 1
    assert (ckpt_dir / "last_model.safetensors").exists()
    assert (ckpt_dir / "training_curves.csv").exists()

    # Sesión 2: Reanudar con 1 época adicional (debe ir de época 2 a 2, acumulando 2 épocas en el CSV)
    rep2 = train_laya(cfg_file, device_name="cpu", dry_run=True, override_epochs=1, override_resume=True)
    assert rep2["hyperparameters"]["resumed_from_weights"] is True
    assert rep2["hyperparameters"]["start_epoch"] == 2
    assert rep2["hyperparameters"]["end_epoch"] == 2
    assert rep2["hyperparameters"]["total_epochs_in_history"] == 2

    # Sesión 3: Reanudar eliminando el archivo CSV (debe partir en 1 pero con last_model como base)
    (ckpt_dir / "training_curves.csv").unlink()
    (ckpt_dir / "training_curves.json").unlink()
    rep3 = train_laya(cfg_file, device_name="cpu", dry_run=True, override_epochs=1, override_resume=True)
    assert rep3["hyperparameters"]["resumed_from_weights"] is True
    assert rep3["hyperparameters"]["start_epoch"] == 1
    assert rep3["hyperparameters"]["end_epoch"] == 1
    assert rep3["hyperparameters"]["total_epochs_in_history"] == 1
