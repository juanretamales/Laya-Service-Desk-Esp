"""Tests unitarios para el harness de evaluación y cálculo de métricas."""

from pathlib import Path
import pytest

from src.evaluation.benchmark import (
    LayaInferenceWrapper,
    compute_classification_metrics,
    run_benchmark_evaluation,
)


def test_compute_metrics_perfect_predictions():
    """Valida métricas perfectas (Accuracy = 1.0, F1 = 1.0)."""
    y_true = ["software", "hardware", "access", "email"]
    y_pred = ["software", "hardware", "access", "email"]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["total_samples"] == 4


def test_compute_metrics_mixed_predictions():
    """Valida el cálculo de métricas y la matriz de confusión con predicciones mixtas."""
    y_true = ["software", "software", "hardware", "hardware"]
    y_pred = ["software", "hardware", "hardware", "hardware"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=["software", "hardware"])
    assert metrics["accuracy"] == 0.75
    assert metrics["total_samples"] == 4

    cm = metrics["confusion_matrix"]
    assert cm["software"]["software"] == 1
    assert cm["software"]["hardware"] == 1
    assert cm["hardware"]["hardware"] == 2
    assert cm["hardware"]["software"] == 0

    assert metrics["per_class"]["hardware"]["recall"] == 1.0
    assert metrics["per_class"]["software"]["recall"] == 0.5


def test_compute_metrics_invalid_inputs():
    """Valida que lance ValueError con entradas inválidas."""
    with pytest.raises(ValueError, match="Discrepancia de longitud"):
        compute_classification_metrics(["software"], ["software", "hardware"])

    with pytest.raises(ValueError, match="secuencias vacías"):
        compute_classification_metrics([], [])


def test_laya_wrapper_missing_package():
    """Valida que el wrapper lance RuntimeError explicativo si laya no está disponible."""
    wrapper = LayaInferenceWrapper()
    with pytest.raises(RuntimeError, match="La biblioteca 'laya' no está instalada"):
        wrapper.load_model()


def test_run_benchmark_evaluation_with_mock(tmp_path: Path):
    """Valida la ejecución completa del benchmark con un predictor mock y exportación de reporte."""
    test_file = tmp_path / "test.jsonl"
    report_file = tmp_path / "reports" / "test_report.json"

    records = [
        {
            "ticket_id": "t1",
            "state": {"subject": "VPN caída", "body": "No conecta"},
            "questions": {"category": {"type": "choice"}},
            "targets": {"category": "network", "priority": "high", "support_group": "network"},
        },
        {
            "ticket_id": "t2",
            "state": {"subject": "Pantalla azul", "body": "Error de memoria"},
            "questions": {"category": {"type": "choice"}},
            "targets": {"category": "hardware", "priority": "medium", "support_group": "workplace"},
        },
    ]

    import json
    with test_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Mock predictor que predice 'network' para t1 y 'hardware' para t2
    def mock_predict(state, questions):
        if "VPN" in state.get("subject", ""):
            return {"category": "network", "priority": "high", "support_group": "network"}
        return {"category": "hardware", "priority": "medium", "support_group": "workplace"}

    report = run_benchmark_evaluation(
        test_file_path=test_file,
        predict_fn=mock_predict,
        tasks=["category", "priority", "support_group"],
        output_report_path=report_file,
        metadata={"model": "mock_test_model"},
    )

    assert report["test_dataset"]["total_records"] == 2
    assert "sha256" in report["test_dataset"]
    assert report["tasks_metrics"]["category"]["accuracy"] == 1.0
    assert report["tasks_metrics"]["priority"]["accuracy"] == 1.0
    assert report["tasks_metrics"]["support_group"]["accuracy"] == 1.0
    assert report["latency_ms"]["mean"] >= 0.0
    assert report_file.exists()
