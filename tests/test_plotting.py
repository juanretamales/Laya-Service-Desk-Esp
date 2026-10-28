"""Tests unitarios para el módulo de curvas de entrenamiento y visualizaciones científicas."""

import json
from pathlib import Path
import pytest

from src.evaluation.plotting import (
    export_training_curves_csv,
    export_training_curves_json,
    load_training_curves,
    plot_all_training_visualizations,
    plot_loss_curve,
    plot_metrics_curves,
    plot_training_dashboard,
)


@pytest.fixture
def sample_history():
    """Fixture con historial simulado de 3 épocas de entrenamiento."""
    return [
        {
            "epoch": 1,
            "train_loss": 1.2500,
            "val_loss": 1.1000,
            "overall_macro_f1": 0.4500,
            "category_macro_f1": 0.4000,
            "category_accuracy": 0.5000,
            "priority_macro_f1": 0.6000,
            "priority_accuracy": 0.6500,
            "support_group_macro_f1": 0.3500,
            "support_group_accuracy": 0.4000,
            "val_metrics": {
                "category": {"macro_f1": 0.4000, "accuracy": 0.5000},
                "priority": {"macro_f1": 0.6000, "accuracy": 0.6500},
                "support_group": {"macro_f1": 0.3500, "accuracy": 0.4000},
            },
        },
        {
            "epoch": 2,
            "train_loss": 0.8500,
            "val_loss": 0.7200,
            "overall_macro_f1": 0.6800,
            "category_macro_f1": 0.6500,
            "category_accuracy": 0.7000,
            "priority_macro_f1": 0.7500,
            "priority_accuracy": 0.8000,
            "support_group_macro_f1": 0.6400,
            "support_group_accuracy": 0.6800,
            "val_metrics": {
                "category": {"macro_f1": 0.6500, "accuracy": 0.7000},
                "priority": {"macro_f1": 0.7500, "accuracy": 0.8000},
                "support_group": {"macro_f1": 0.6400, "accuracy": 0.6800},
            },
        },
        {
            "epoch": 3,
            "train_loss": 0.6000,
            "val_loss": 0.7500,
            "overall_macro_f1": 0.6600,
            "category_macro_f1": 0.6300,
            "category_accuracy": 0.6800,
            "priority_macro_f1": 0.7400,
            "priority_accuracy": 0.7900,
            "support_group_macro_f1": 0.6100,
            "support_group_accuracy": 0.6500,
            "val_metrics": {
                "category": {"macro_f1": 0.6300, "accuracy": 0.6800},
                "priority": {"macro_f1": 0.7400, "accuracy": 0.7900},
                "support_group": {"macro_f1": 0.6100, "accuracy": 0.6500},
            },
        },
    ]


def test_export_curves_csv_and_json(tmp_path: Path, sample_history):
    """Valida la exportación correcta tanto en CSV como en JSON."""
    csv_file = tmp_path / "curves.csv"
    json_file = tmp_path / "curves.json"

    export_training_curves_csv(sample_history, csv_file)
    export_training_curves_json(sample_history, json_file)

    assert csv_file.exists()
    assert json_file.exists()

    content = csv_file.read_text(encoding="utf-8")
    assert "epoch,train_loss,val_loss,overall_macro_f1" in content
    assert "1,1.25,1.1,0.45" in content

    loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert len(loaded_json) == 3
    assert loaded_json[0]["epoch"] == 1


def test_load_training_curves(tmp_path: Path, sample_history):
    """Valida la carga y normalización desde memoria, JSON, CSV y reporte completo."""
    # Desde lista directa
    recs_from_list = load_training_curves(sample_history)
    assert len(recs_from_list) == 3

    # Desde archivo JSON de lista
    json_file = tmp_path / "curves.json"
    export_training_curves_json(sample_history, json_file)
    recs_from_json = load_training_curves(json_file)
    assert len(recs_from_json) == 3

    # Desde training_report.json
    report_file = tmp_path / "training_report.json"
    report_file.write_text(json.dumps({"history": sample_history}), encoding="utf-8")
    recs_from_report = load_training_curves(report_file)
    assert len(recs_from_report) == 3

    # Desde CSV
    csv_file = tmp_path / "curves.csv"
    export_training_curves_csv(sample_history, csv_file)
    recs_from_csv = load_training_curves(csv_file)
    assert len(recs_from_csv) == 3
    assert recs_from_csv[1]["epoch"] == 2
    assert abs(recs_from_csv[1]["train_loss"] - 0.85) < 1e-4


def test_load_training_curves_invalid(tmp_path: Path):
    """Valida errores apropiados ante archivos inexistentes o formatos inválidos."""
    with pytest.raises(FileNotFoundError):
        load_training_curves(tmp_path / "missing.json")

    invalid_txt = tmp_path / "data.txt"
    invalid_txt.write_text("random text", encoding="utf-8")
    with pytest.raises(ValueError, match="Extensión no soportada"):
        load_training_curves(invalid_txt)


def test_plot_loss_curve(tmp_path: Path, sample_history):
    """Valida la generación de la curva de pérdida en imagen PNG."""
    out_png = tmp_path / "loss.png"
    result_path = plot_loss_curve(sample_history, out_png, best_epoch=2)

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Archivo binario con contenido visual


def test_plot_metrics_curves(tmp_path: Path, sample_history):
    """Valida la generación de la curva de métricas multitarea."""
    out_png = tmp_path / "metrics.png"
    result_path = plot_metrics_curves(
        sample_history,
        out_png,
        tasks=["category", "priority", "support_group"],
        best_epoch=2,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000


def test_plot_training_dashboard(tmp_path: Path, sample_history):
    """Valida la generación del dashboard unificado de entrenamiento."""
    out_png = tmp_path / "dashboard.png"
    result_path = plot_training_dashboard(
        sample_history,
        out_png,
        tasks=["category", "priority", "support_group"],
        best_epoch=2,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000


def test_plot_all_training_visualizations(tmp_path: Path, sample_history):
    """Valida la generación coordinada de todas las visualizaciones."""
    out_dir = tmp_path / "figures"
    results = plot_all_training_visualizations(
        sample_history,
        out_dir,
        tasks=["category", "priority", "support_group"],
        best_epoch=2,
    )

    assert "loss_curve" in results
    assert "metrics_curve" in results
    assert "dashboard" in results
    assert results["loss_curve"].exists()
    assert results["metrics_curve"].exists()
    assert results["dashboard"].exists()


def test_empty_history_errors(tmp_path: Path):
    """Valida que un historial vacío lance ValueError para proteger la integridad."""
    with pytest.raises(ValueError, match="está vacío"):
        export_training_curves_csv([], tmp_path / "empty.csv")

    with pytest.raises(ValueError, match="está vacío"):
        export_training_curves_json([], tmp_path / "empty.json")

    with pytest.raises(ValueError, match="Historial vacío"):
        plot_loss_curve([], tmp_path / "empty.png")
