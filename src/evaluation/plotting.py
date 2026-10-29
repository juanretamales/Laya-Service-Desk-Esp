"""Módulo de generación de gráficos y visualización científica de curvas de entrenamiento.

Este módulo implementa la exportación tabular (CSV/JSON) y la renderización de figuras
de calidad de publicación (300 DPI, estilo académico, rejillas sutiles, paletas accesibles)
para monitorear y documentar la evolución de la pérdida (Loss) y las métricas de rendimiento
(Accuracy, Macro-F1) a lo largo de las épocas de entrenamiento y validación de Laya.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Backend headless seguro para entornos sin servidor X (CI, Colab, terminal)
import matplotlib.pyplot as plt  # noqa: E402


# Paleta de colores sobria y accesible (estilo paper científico)
COLOR_TRAIN = "#1f77b4"      # Azul clásico
COLOR_VAL = "#d62728"        # Rojo / Carmesí
COLOR_CATEGORY = "#2ca02c"   # Verde
COLOR_PRIORITY = "#ff7f0e"   # Naranja
COLOR_SUPPORT = "#9467bd"    # Púrpura
COLOR_OVERALL = "#17becf"    # Cian / Azul turquesa
COLOR_BEST_LINE = "#7f7f7f"  # Gris neutro


def export_training_curves_csv(
    history: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Exporta el historial de épocas a un archivo CSV estructurado para análisis tabular.

    Args:
        history: Lista de diccionarios de métricas por época.
        output_path: Ruta destino del archivo CSV.

    Returns:
        Ruta al archivo CSV creado.

    Raises:
        ValueError: Si el historial está vacío.
    """
    if not history:
        raise ValueError("El historial de entrenamiento está vacío; no se puede exportar a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Identificar todas las columnas disponibles preservando el orden canónico
    standard_cols = ["epoch", "train_loss", "val_loss", "overall_macro_f1"]
    dynamic_cols = []
    for entry in history:
        for k in entry.keys():
            if k not in standard_cols and k != "val_metrics" and k not in dynamic_cols:
                dynamic_cols.append(k)

    fieldnames = standard_cols + dynamic_cols

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            flat_row = dict(row)
            # Aplanar val_metrics si no estuvieran aplanadas previamente
            if "val_metrics" in row and isinstance(row["val_metrics"], dict):
                for task, metrics in row["val_metrics"].items():
                    if isinstance(metrics, dict):
                        for m_key, m_val in metrics.items():
                            col_name = f"{task}_{m_key}"
                            if col_name not in flat_row:
                                flat_row[col_name] = m_val
            writer.writerow(flat_row)

    return output_path


def export_training_curves_json(
    history: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Exporta el historial de curvas de entrenamiento a un archivo JSON legible.

    Args:
        history: Lista de diccionarios de métricas por época.
        output_path: Ruta destino del archivo JSON.

    Returns:
        Ruta al archivo JSON creado.

    Raises:
        ValueError: Si el historial está vacío.
    """
    if not history:
        raise ValueError("El historial de entrenamiento está vacío; no se puede exportar a JSON.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return output_path


def load_training_curves(
    source: Union[List[Dict[str, Any]], Path, str],
) -> List[Dict[str, Any]]:
    """Carga y normaliza el historial de entrenamiento desde memoria o archivo (JSON/CSV).

    Args:
        source: Lista de diccionarios, o ruta hacia archivo JSON o CSV o training_report.json.

    Returns:
        Lista de diccionarios normalizados con métricas por época.

    Raises:
        FileNotFoundError: Si la ruta especificada no existe.
        ValueError: Si el formato del archivo o contenido no es compatible.
    """
    if isinstance(source, list):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de curvas no encontrado: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "history" in data and isinstance(data["history"], list):
            return data["history"]
        raise ValueError(f"Formato JSON inesperado en {path}: no contiene una lista de épocas ni campo 'history'.")

    elif path.suffix.lower() == ".csv":
        records = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row: Dict[str, Any] = {}
                for k, v in row.items():
                    if k == "epoch":
                        parsed_row[k] = int(v)
                    else:
                        try:
                            parsed_row[k] = float(v)
                        except (ValueError, TypeError):
                            parsed_row[k] = v
                records.append(parsed_row)
        return records

    raise ValueError(f"Extensión no soportada para curvas: {path.suffix}. Use .json o .csv.")


def plot_loss_curve(
    history: List[Dict[str, Any]],
    output_path: Path,
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    """Genera y guarda el gráfico de curvas de pérdida (Train Loss vs Val Loss) por época.

    Args:
        history: Lista de métricas registradas en cada época.
        output_path: Ruta destino de la imagen (PNG).
        best_epoch: Época seleccionada como mejor modelo (para marcar con línea vertical).
        title: Título personalizado para el gráfico.

    Returns:
        Ruta al archivo de imagen generado.
    """
    if not history:
        raise ValueError("Historial vacío: no se puede graficar la curva de pérdida.")

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h.get("train_loss", 0.0)) for h in history]
    val_loss = [float(h.get("val_loss", 0.0)) for h in history]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color=COLOR_TRAIN, label="Pérdida Entrenamiento (Train)")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, color=COLOR_VAL, label="Pérdida Validación (Val)")

    if best_epoch is not None and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        best_val = val_loss[idx]
        ax.axvline(x=best_epoch, color=COLOR_BEST_LINE, linestyle="--", linewidth=1.5,
                   label=f"Mejor Época ({best_epoch})")
        ax.scatter([best_epoch], [best_val], color=COLOR_VAL, s=100, zorder=5)
        ax.annotate(
            f"Época {best_epoch}\nVal Loss: {best_val:.4f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.1, best_val + 0.05 * (max(val_loss) - min(val_loss) + 0.01)),
            arrowprops=dict(arrowstyle="->", color=COLOR_BEST_LINE, lw=1.2),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_BEST_LINE, alpha=0.9),
        )

    ax.set_title(title or "Evolución de la Pérdida de Entrenamiento y Validación - Laya Service Desk",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Época de Entrenamiento", fontsize=10, fontweight="bold")
    ax.set_ylabel("Pérdida (Cross-Entropy Loss)", fontsize=10, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_metrics_curves(
    history: List[Dict[str, Any]],
    output_path: Path,
    tasks: Optional[List[str]] = None,
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    """Genera y guarda el gráfico de curvas de métricas (Macro-F1 y Accuracy) por época.

    Args:
        history: Lista de métricas registradas en cada época.
        output_path: Ruta destino de la imagen (PNG).
        tasks: Lista de tareas a graficar (default: ['category', 'priority', 'support_group']).
        best_epoch: Época seleccionada como mejor modelo.
        title: Título personalizado para el gráfico.

    Returns:
        Ruta al archivo de imagen generado.
    """
    if not history:
        raise ValueError("Historial vacío: no se puede graficar las métricas.")

    if tasks is None:
        tasks = ["category", "priority", "support_group"]

    epochs = [int(h["epoch"]) for h in history]
    overall_f1 = [float(h.get("overall_macro_f1", 0.0)) for h in history]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    task_colors = {
        "category": COLOR_CATEGORY,
        "priority": COLOR_PRIORITY,
        "support_group": COLOR_SUPPORT,
    }
    task_markers = {
        "category": "^",
        "priority": "D",
        "support_group": "v",
    }

    # Curvas por tarea
    for task in tasks:
        task_f1 = []
        for h in history:
            val_m = h.get("val_metrics", {})
            f1 = val_m.get(task, {}).get("macro_f1")
            if f1 is None:
                f1 = h.get(f"{task}_macro_f1", 0.0)
            task_f1.append(float(f1))

        color = task_colors.get(task, "#333333")
        marker = task_markers.get(task, "o")
        ax.plot(epochs, task_f1, marker=marker, linewidth=1.8, linestyle="-.",
                color=color, label=f"Macro-F1 {task.capitalize()}")

    # Curva general Macro-F1 promedio
    ax.plot(epochs, overall_f1, marker="o", linewidth=2.8, color=COLOR_OVERALL,
            label="Macro-F1 Global (Promedio)")

    if best_epoch is not None and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        best_f1 = overall_f1[idx]
        ax.axvline(x=best_epoch, color=COLOR_BEST_LINE, linestyle="--", linewidth=1.5,
                   label=f"Mejor Época ({best_epoch})")
        ax.scatter([best_epoch], [best_f1], color=COLOR_OVERALL, s=120, zorder=5)
        ax.annotate(
            f"Mejor Macro-F1: {best_f1:.4f}\n(Época {best_epoch})",
            xy=(best_epoch, best_f1),
            xytext=(best_epoch + 0.1, best_f1 - 0.04),
            arrowprops=dict(arrowstyle="->", color=COLOR_BEST_LINE, lw=1.2),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_BEST_LINE, alpha=0.9),
        )

    ax.set_title(title or "Evolución de Rendimiento Multitarea (Validation Macro-F1) - Laya",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Época de Entrenamiento", fontsize=10, fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_training_dashboard(
    history: List[Dict[str, Any]],
    output_path: Path,
    tasks: Optional[List[str]] = None,
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    """Genera un dashboard visual integrado de 2 paneles (Pérdida y Métricas) listo para publicación.

    Args:
        history: Lista de métricas registradas en cada época.
        output_path: Ruta destino de la imagen (PNG).
        tasks: Lista de tareas a incluir en el panel de métricas.
        best_epoch: Época seleccionada como mejor modelo.
        title: Título global del dashboard.

    Returns:
        Ruta al archivo de imagen generado.
    """
    if not history:
        raise ValueError("Historial vacío: no se puede generar el dashboard de entrenamiento.")

    if tasks is None:
        tasks = ["category", "priority", "support_group"]

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h.get("train_loss", 0.0)) for h in history]
    val_loss = [float(h.get("val_loss", 0.0)) for h in history]
    overall_f1 = [float(h.get("overall_macro_f1", 0.0)) for h in history]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Panel Izquierdo: Pérdida
    ax_loss.plot(epochs, train_loss, marker="o", linewidth=2.2, color=COLOR_TRAIN, label="Train Loss")
    ax_loss.plot(epochs, val_loss, marker="s", linewidth=2.2, color=COLOR_VAL, label="Val Loss")
    if best_epoch is not None and best_epoch in epochs:
        ax_loss.axvline(x=best_epoch, color=COLOR_BEST_LINE, linestyle="--", linewidth=1.5,
                        label=f"Mejor Época ({best_epoch})")
    ax_loss.set_title("Curvas de Pérdida (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Época", fontsize=10, fontweight="bold")
    ax_loss.set_ylabel("Pérdida", fontsize=10, fontweight="bold")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(frameon=True, fontsize=9, loc="upper right")

    # Panel Derecho: Métricas Multitarea
    task_colors = {
        "category": COLOR_CATEGORY,
        "priority": COLOR_PRIORITY,
        "support_group": COLOR_SUPPORT,
    }
    for task in tasks:
        task_f1 = []
        for h in history:
            val_m = h.get("val_metrics", {})
            f1 = val_m.get(task, {}).get("macro_f1")
            if f1 is None:
                f1 = h.get(f"{task}_macro_f1", 0.0)
            task_f1.append(float(f1))
        ax_f1.plot(epochs, task_f1, marker="^", linewidth=1.6, linestyle="-.",
                   color=task_colors.get(task, "#333"), label=f"F1 {task.capitalize()}")

    ax_f1.plot(epochs, overall_f1, marker="o", linewidth=2.8, color=COLOR_OVERALL, label="F1 Global Promedio")
    if best_epoch is not None and best_epoch in epochs:
        ax_f1.axvline(x=best_epoch, color=COLOR_BEST_LINE, linestyle="--", linewidth=1.5,
                      label=f"Mejor Época ({best_epoch})")
    ax_f1.set_title("Evolución de Macro-F1 por Tarea", fontsize=11, fontweight="bold")
    ax_f1.set_xlabel("Época", fontsize=10, fontweight="bold")
    ax_f1.set_ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    ax_f1.set_xticks(epochs)
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.grid(True, linestyle=":", alpha=0.6)
    ax_f1.legend(frameon=True, fontsize=9, loc="lower right")

    fig.suptitle(title or "Panel Integral de Rendimiento: Fine-Tuning Laya Multilingual en Service Desk",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_all_training_visualizations(
    history: List[Dict[str, Any]],
    output_dir: Path,
    tasks: Optional[List[str]] = None,
    best_epoch: Optional[int] = None,
) -> Dict[str, Path]:
    """Genera de manera coordinada todos los gráficos y reportes visuales de entrenamiento.

    Args:
        history: Lista de métricas registradas en cada época.
        output_dir: Directorio donde se guardarán las imágenes.
        tasks: Lista de tareas a evaluar.
        best_epoch: Época seleccionada como mejor modelo.

    Returns:
        Diccionario con las rutas de las imágenes generadas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = output_dir / "training_loss_curve.png"
    metrics_path = output_dir / "training_metrics_curve.png"
    dashboard_path = output_dir / "training_dashboard.png"

    plot_loss_curve(history, loss_path, best_epoch=best_epoch)
    plot_metrics_curves(history, metrics_path, tasks=tasks, best_epoch=best_epoch)
    plot_training_dashboard(history, dashboard_path, tasks=tasks, best_epoch=best_epoch)

    return {
        "loss_curve": loss_path,
        "metrics_curve": metrics_path,
        "dashboard": dashboard_path,
    }
