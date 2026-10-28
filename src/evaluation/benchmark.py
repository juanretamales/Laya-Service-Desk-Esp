"""Harness de evaluación científica y benchmark para modelos Laya.

Este módulo implementa el protocolo de evaluación zero-shot y fine-tuned sobre
el conjunto de prueba congelado (Gate G3/G5). Calcula métricas estadísticas
estrictas: Accuracy, F1-macro, F1-micro, desglose por clase, matrices de confusión
y latencias de inferencia.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set

from src.preprocessing.split import calculate_file_sha256
from src.preprocessing.unify import (
    CANONICAL_CATEGORIES,
    CANONICAL_PRIORITIES,
    CANONICAL_SUPPORT_GROUPS,
)


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calcula métricas de clasificación multicapa: Accuracy, Macro/Micro F1, Precision, Recall.

    Args:
        y_true: Lista de etiquetas verdaderas (ground truth).
        y_pred: Lista de etiquetas predichas por el modelo.
        labels: Lista explícita de etiquetas canónicas a considerar.

    Returns:
        Diccionario exhaustivo con accuracy, macro/micro F1, desglose por clase y matriz de confusión.

    Raises:
        ValueError: Si las listas tienen longitudes diferentes o están vacías.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Discrepancia de longitud entre y_true ({len(y_true)}) y y_pred ({len(y_pred)})"
        )
    if not y_true:
        raise ValueError("No se pueden calcular métricas sobre secuencias vacías")

    all_labels_set: Set[str] = set(y_true).union(set(y_pred))
    if labels:
        target_labels = list(dict.fromkeys(labels + [label for label in sorted(all_labels_set) if label not in labels]))
    else:
        target_labels = sorted(all_labels_set)

    total_samples = len(y_true)
    correct_samples = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_samples / total_samples

    # Matriz de confusión: {clase_verdadera: {clase_predicha: conteo}}
    confusion_matrix: Dict[str, Dict[str, int]] = {
        label: {pred_label: 0 for pred_label in target_labels} for label in target_labels
    }
    for yt, yp in zip(y_true, y_pred):
        if yt in confusion_matrix and yp in confusion_matrix[yt]:
            confusion_matrix[yt][yp] += 1

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    global_tp = 0
    global_fp = 0
    global_fn = 0

    for label in target_labels:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in target_labels if other != label)
        fn = sum(confusion_matrix[label][other] for other in target_labels if other != label)
        support = sum(confusion_matrix[label].values())

        global_tp += tp
        global_fp += fp
        global_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        # Solo computar para el promedio macro si la clase tiene soporte real
        if support > 0:
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # Micro F1
    micro_denominator = global_tp + 0.5 * (global_fp + global_fn)
    micro_f1 = global_tp / micro_denominator if micro_denominator > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "total_samples": total_samples,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix,
    }


class LayaInferenceWrapper:
    """Wrapper adaptador para inferencia con modelos Laya (laya.Router)."""

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._router: Optional[Any] = None

    def load_model(self) -> None:
        """Carga el modelo Router de Laya."""
        try:
            from laya import Router  # type: ignore
            # Si model_id se especifica y difiere de default, se pasa; en otro caso se usa device
            self._router = Router(device=self.device)
        except ImportError as e:
            raise RuntimeError(
                f"La biblioteca 'laya' no está instalada en el entorno actual. "
                f"Para inferencia real instale el paquete o ejecute en Colab: {e}"
            ) from e

    def predict(self, state: Dict[str, str], questions: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Ejecuta inferencia multi-tarea sobre un estado de ticket y extrae las decisiones elegidas.

        Args:
            state: Diccionario con subject y body del ticket.
            questions: Diccionario de preguntas tipadas ('choice').

        Returns:
            Diccionario de respuestas predichas por tarea (ej: {'category': 'software'}).
        """
        if self._router is None:
            self.load_model()

        # Adaptar esquema de preguntas para el motor de Laya (instructions + criteria)
        laya_q: Dict[str, Dict[str, Any]] = {}
        for qid, qdef in questions.items():
            q_type = qdef.get("type", "choice")
            instructions = qdef.get("instructions") or qdef.get("question") or "¿Cuál es la respuesta?"
            criteria = qdef.get("criteria") or qdef.get("options") or []
            laya_q[qid] = {
                "type": q_type,
                "instructions": instructions,
                "criteria": criteria,
            }

        res = self._router.predict(state, laya_q)
        answers = res.get("answers", {}) if isinstance(res, dict) else {}

        predictions: Dict[str, str] = {}
        for qid in questions.keys():
            ans_item = answers.get(qid, {})
            if isinstance(ans_item, dict):
                predictions[qid] = str(ans_item.get("choice", ans_item.get("score", "unknown")))
            else:
                predictions[qid] = str(ans_item)

        return predictions


def run_benchmark_evaluation(
    test_file_path: Path,
    predict_fn: Callable[[Dict[str, str], Dict[str, Dict[str, Any]]], Dict[str, str]],
    tasks: Optional[List[str]] = None,
    output_report_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ejecuta la evaluación sobre el archivo de test congelado midiendo métricas y latencia.

    Args:
        test_file_path: Ruta al archivo test.jsonl congelado.
        predict_fn: Función callable de inferencia (state, questions) -> dict de predicciones.
        tasks: Lista de tareas a evaluar (por defecto ['category', 'priority', 'support_group']).
        output_report_path: Ruta opcional donde guardar el reporte estructurado en JSON.
        metadata: Metadatos adicionales para documentar el experimento en el reporte.

    Returns:
        Diccionario estructurado con el reporte completo de la evaluación.
    """
    if not test_file_path.exists():
        raise FileNotFoundError(f"Archivo de prueba no encontrado en: {test_file_path}")

    active_tasks = tasks or ["category", "priority", "support_group"]
    file_sha256 = calculate_file_sha256(test_file_path)

    records: List[Dict[str, Any]] = []
    with test_file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    if not records:
        raise ValueError(f"El archivo de prueba {test_file_path} está vacío")

    ground_truth: Dict[str, List[str]] = defaultdict(list)
    predictions: Dict[str, List[str]] = defaultdict(list)
    latencies_ms: List[float] = []

    for item in records:
        state = item.get("state", {})
        questions = item.get("questions", {})
        targets = item.get("targets", {})

        # Inferencia cronometrada
        start_t = time.perf_counter()
        pred_dict = predict_fn(state, questions)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies_ms.append(elapsed_ms)

        for task in active_tasks:
            true_label = str(targets.get(task, "unknown"))
            predicted_label = str(pred_dict.get(task, "unknown"))
            ground_truth[task].append(true_label)
            predictions[task].append(predicted_label)

    # Cálculo de métricas por tarea
    canonical_options_map = {
        "category": list(CANONICAL_CATEGORIES),
        "priority": list(CANONICAL_PRIORITIES),
        "support_group": list(CANONICAL_SUPPORT_GROUPS),
    }

    task_metrics: Dict[str, Any] = {}
    for task in active_tasks:
        labels = canonical_options_map.get(task)
        task_metrics[task] = compute_classification_metrics(
            y_true=ground_truth[task],
            y_pred=predictions[task],
            labels=labels,
        )

    # Resumen de latencias
    latencies_ms.sort()
    n_samples = len(latencies_ms)
    mean_latency = sum(latencies_ms) / n_samples
    p50_latency = latencies_ms[int(n_samples * 0.50)]
    p95_latency = latencies_ms[int(n_samples * 0.95)]

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_dataset": {
            "path": str(test_file_path),
            "sha256": file_sha256,
            "total_records": n_samples,
        },
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
            "total_seconds": round(sum(latencies_ms) / 1000.0, 3),
        },
        "tasks_metrics": task_metrics,
        "metadata": metadata or {},
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with output_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    """Punto de entrada CLI para ejecutar la evaluación de baseline o modelos fine-tuned."""
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar benchmark de evaluación sobre test set congelado")
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/splits/test.jsonl",
        help="Ruta al archivo test.jsonl congelado",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/baseline_results.json",
        help="Ruta de salida para el reporte de métricas JSON",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="convaiinnovations/laya-multilingual",
        help="Identificador del modelo en HuggingFace o ruta local",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de cómputo para inferencia",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecutar con predictor dummy/fallback para pruebas sin GPU o sin laya instalado",
    )
    args = parser.parse_args()

    test_path = Path(args.test_file)
    output_path = Path(args.output_report)

    if args.dry_run:
        print("[AVISO] Ejecutando benchmark en modo --dry-run con predictor determinista fallback...")

        def dry_run_predict(state: Dict[str, str], questions: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
            return {"category": "other", "priority": "unknown", "support_group": "unknown"}

        predict_fn = dry_run_predict
    else:
        print(f"Cargando Laya Router con modelo: {args.model_id} en {args.device}...")
        wrapper = LayaInferenceWrapper(model_id=args.model_id, device=args.device)
        predict_fn = wrapper.predict

    report = run_benchmark_evaluation(
        test_file_path=test_path,
        predict_fn=predict_fn,
        output_report_path=output_path,
        metadata={
            "model_id": args.model_id,
            "device": args.device,
            "dry_run": args.dry_run,
        },
    )

    print("\n--- RESUMEN DE EVALUACIÓN ---")
    print(f"Test Set Evaluado: {report['test_dataset']['path']} ({report['test_dataset']['total_records']} tickets)")
    print(f"SHA-256 Test Set:  {report['test_dataset']['sha256']}")
    print(f"Latencia Media:    {report['latency_ms']['mean']} ms/ticket")
    for task_name, task_data in report["tasks_metrics"].items():
        print(f"Tarea '{task_name}': Accuracy={task_data['accuracy']:.4f}, Macro-F1={task_data['macro_f1']:.4f}")
    print(f"Reporte exportado exitosamente en: {output_path}")


if __name__ == "__main__":
    main()
