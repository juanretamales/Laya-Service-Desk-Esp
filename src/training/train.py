"""Módulo de entrenamiento y fine-tuning para Laya Multilingual en Service Desk.

Este módulo implementa el pipeline de ajuste supervisado respetando el protocolo RLCD
sobre las preguntas tipadas ('choice') para las tareas del Service Desk (category, priority, support_group).
Garantiza aislamiento estricto: entrena únicamente con train.jsonl y valida con val.jsonl.
El conjunto de prueba congelado (test.jsonl) nunca es consultado durante este proceso.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from src.evaluation.benchmark import compute_classification_metrics
from src.evaluation.plotting import (
    export_training_curves_csv,
    export_training_curves_json,
    plot_all_training_visualizations,
)
from src.preprocessing.split import calculate_file_sha256, verify_zero_leakage


def set_seed(seed: int = 42) -> None:
    """Fija las semillas pseudoaleatorias para reproducibilidad estricta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_config(config_path: Path) -> Dict[str, Any]:
    """Carga y valida el archivo de configuración YAML de entrenamiento.

    Args:
        config_path: Ruta al archivo YAML (ej: configs/training.yaml).

    Returns:
        Diccionario con la configuración validada.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan secciones requeridas o se intenta apuntar al test set.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validaciones de seguridad de splits: prohibir referenciar el test set
    train_stem = Path(cfg.get("data", {}).get("train_file", "")).stem.lower()
    val_stem = Path(cfg.get("data", {}).get("val_file", "")).stem.lower()

    if "test" in train_stem or "test" in val_stem:
        raise ValueError(
            "Violación de protocolo de aislamiento: Ni train_file ni val_file pueden referenciar el test set"
        )

    return cfg


def load_jsonl_records(file_path: Path) -> List[Dict[str, Any]]:
    """Carga registros de un archivo JSONL."""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def encode_record_items(
    record: Dict[str, Any],
    agent: Any,
    tasks: List[str],
) -> List[Dict[str, Any]]:
    """Codifica un registro y sus preguntas tipadas, etiquetando los targets de supervisión.

    Args:
        record: Registro de ticket con state, questions y targets.
        agent: Instancia de laya.Agent.
        tasks: Lista de nombres de tareas a incluir (ej: ['category', 'priority', 'support_group']).

    Returns:
        Lista de items procesados listos para collate_items.
    """
    from laya.agent import Agent  # type: ignore

    state = record.get("state", {})
    questions = record.get("questions", {})
    targets = record.get("targets", {})

    active_qids = [t for t in tasks if t in questions]
    if not active_qids:
        return []

    # Normalizar preguntas al formato interno de Laya
    internal: Dict[str, Dict[str, Any]] = {}
    for qid in active_qids:
        qdef = questions[qid]
        laya_qdef = {
            "type": qdef.get("type", "choice"),
            "instructions": qdef.get("instructions") or qdef.get("question") or "¿Cuál es la respuesta?",
            "criteria": qdef.get("criteria") or qdef.get("options") or [],
        }
        internal[qid] = Agent._to_internal(laya_qdef)

    items = agent._encode_state(state, active_qids, internal)

    # Anotar etiquetas supervisadas en cada item codificado
    valid_items = []
    for qid, item in zip(active_qids, items):
        if qid not in internal:
            continue

        item["id"] = qid
        crit_keys = list(internal[qid]["crit"].keys())
        item["crit_keys"] = crit_keys
        true_label = str(targets.get(qid, "unknown"))

        if true_label in crit_keys:
            target_idx = crit_keys.index(true_label)
            num_options = len(item["markers"])
            target_dist = np.zeros(num_options, dtype=np.float32)
            if target_idx < num_options:
                target_dist[target_idx] = 1.0
                item["label"] = target_idx
                item["target"] = target_dist
                valid_items.append(item)

    return valid_items


def evaluate_validation(
    agent: Any,
    val_records: List[Dict[str, Any]],
    tasks: List[str],
    device: torch.device,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evalúa la pérdida y métricas de clasificación sobre el conjunto de validación.

    Args:
        agent: Instancia de laya.Agent.
        val_records: Lista de registros del conjunto de validación.
        tasks: Tareas evaluadas.
        device: Dispositivo de cómputo (CPU/CUDA).

    Returns:
        Tupla con (val_loss_media, metricas_por_tarea).
    """
    from laya.agent import collate_items  # type: ignore

    agent.model.eval()
    total_loss = 0.0
    total_batches = 0

    predictions_by_task: Dict[str, List[str]] = {t: [] for t in tasks}
    targets_by_task: Dict[str, List[str]] = {t: [] for t in tasks}

    with torch.no_grad():
        for record in val_records:
            items = encode_record_items(record, agent, tasks)
            if not items:
                continue

            batch = collate_items([items], agent.tok.pad_token_id)
            if batch is None:
                continue

            fwd_args = {
                k: v.to(device) for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
                         "position_ids", "option_ids"]
            }
            labels = batch["label"].to(device)

            use_cuda_eval = device.type == "cuda"
            autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda_eval else contextlib.nullcontext()
            with autocast_ctx:
                logits, _ = agent.model(**fwd_args)
                loss = F.cross_entropy(logits, labels)

            total_loss += float(loss.item())
            total_batches += 1

            # Extraer predicciones
            pred_indices = logits.argmax(dim=-1).cpu().numpy()
            for i, it in enumerate(items):
                qid = it.get("id")
                crit_keys = it.get("crit_keys", [])
                pred_idx = pred_indices[i]
                true_lbl = str(record.get("targets", {}).get(qid, "unknown"))
                pred_lbl = crit_keys[pred_idx] if crit_keys and pred_idx < len(crit_keys) else "unknown"

                if qid in predictions_by_task:
                    predictions_by_task[qid].append(pred_lbl)
                    targets_by_task[qid].append(true_lbl)

    mean_loss = total_loss / max(1, total_batches)

    task_metrics: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        y_true = targets_by_task[task]
        y_pred = predictions_by_task[task]
        if y_true:
            m = compute_classification_metrics(y_true, y_pred)
            task_metrics[task] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
            }
        else:
            task_metrics[task] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    return mean_loss, task_metrics


def train_laya(
    config_path: Path,
    device_name: str = "cpu",
    dry_run: bool = False,
    override_epochs: Optional[int] = None,
    override_patience: Optional[int] = None,
) -> Dict[str, Any]:
    """Ejecuta el protocolo de fine-tuning para Laya Multilingual con early stopping.

    Args:
        config_path: Ruta a configs/training.yaml.
        device_name: Dispositivo ('cpu' o 'cuda').
        dry_run: Si es True, ejecuta solo un paso mínimo de prueba sin entrenamiento completo.
        override_epochs: Sobrescribe el número de épocas para pruebas rápidas.
        override_patience: Sobrescribe la paciencia de early stopping.

    Returns:
        Diccionario con el reporte final de entrenamiento y metadatos de checkpoints (best y last).
    """
    from laya import Router  # type: ignore
    from laya.agent import collate_items  # type: ignore

    cfg = load_training_config(config_path)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    train_path = Path(cfg["data"]["train_file"])
    val_path = Path(cfg["data"]["val_file"])
    checkpoint_dir = Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints/laya_service_desk"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl_records(train_path)
    val_records = load_jsonl_records(val_path)

    # Validar aislamiento estricto sin solapamiento
    verify_zero_leakage(train_records, val_records, [], hash_key="content_hash")
    train_sha256 = calculate_file_sha256(train_path)
    val_sha256 = calculate_file_sha256(val_path)

    epochs = override_epochs if override_epochs is not None else int(cfg.get("training", {}).get("epochs", 5))
    batch_size = int(cfg.get("training", {}).get("batch_size", 16))
    lr_enc = float(cfg.get("training", {}).get("learning_rate_encoder", 2e-5))
    lr_head = float(cfg.get("training", {}).get("learning_rate_head", 1e-4))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.01))
    tasks = list(cfg.get("data", {}).get("tasks", ["category", "priority", "support_group"]))

    # Configuración de Early Stopping
    patience = override_patience if override_patience is not None else int(
        cfg.get("training", {}).get("early_stopping_patience", 3)
    )
    metric_target_name = str(cfg.get("output", {}).get("save_best_metric", "overall_macro_f1"))

    if dry_run:
        print("[AVISO] Modo --dry-run activado: usando subconjunto mínimo para prueba...")
        train_records = train_records[:4]
        val_records = val_records[:2]
        if override_epochs is None:
            epochs = 1
        batch_size = 2

    target_device = torch.device(device_name)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        print("[AVISO] CUDA no disponible; usando CPU como fallback.")
        target_device = torch.device("cpu")

    print(f"Inicializando Router de Laya en dispositivo: {target_device}...")
    router = Router(device=str(target_device))
    agent = router.load("multilingual")

    # Separar parámetros del encoder y del decision head para tasa de aprendizaje diferencial
    encoder_params = list(agent.model.encoder.parameters())
    head_params = [
        p for n, p in agent.model.named_parameters() if not n.startswith("encoder.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_enc},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    use_cuda = target_device.type == "cuda"
    if use_cuda:
        print("[GPU] Optimizando VRAM: activando gradient checkpointing y FP16...")
        agent.model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        agent.model.head_checkpointing = True

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history: List[Dict[str, Any]] = []

    best_weights_path = checkpoint_dir / "best_model.safetensors"
    last_weights_path = checkpoint_dir / "last_model.safetensors"
    default_model_path = checkpoint_dir / "model.safetensors"

    print(
        f"Iniciando fine-tuning: {len(train_records)} registros de entrenamiento, "
        f"{epochs} épocas (Early Stopping Paciencia: {patience})..."
    )
    start_time = datetime.now(timezone.utc)

    for epoch in range(1, epochs + 1):
        agent.model.train()
        random.shuffle(train_records)
        epoch_loss = 0.0
        batches = 0

        # Iterar en batches de tickets
        for i in range(0, len(train_records), batch_size):
            batch_recs = train_records[i:i + batch_size]
            batch_items = []
            for rec in batch_recs:
                items = encode_record_items(rec, agent, tasks)
                if items:
                    batch_items.extend(items)

            if not batch_items:
                continue

            collated = collate_items([batch_items], agent.tok.pad_token_id)
            if collated is None:
                continue

            fwd_args = {
                k: v.to(target_device) for k, v in collated.items()
                if k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
                         "position_ids", "option_ids"]
            }
            labels = collated["label"].to(target_device)

            optimizer.zero_grad()
            autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext()
            with autocast_ctx:
                logits, _ = agent.model(**fwd_args)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            batches += 1

        avg_train_loss = epoch_loss / max(1, batches)

        # Evaluación completa de validación en cada época
        val_loss, val_metrics = evaluate_validation(agent, val_records, tasks, target_device)

        # Calcular macro-F1 global promedio entre tareas
        macro_f1_sum = sum(val_metrics.get(t, {}).get("macro_f1", 0.0) for t in tasks)
        overall_macro_f1 = macro_f1_sum / max(1, len(tasks))

        # Determinar métrica objetivo para early stopping
        if metric_target_name == "category_macro_f1":
            current_eval_metric = val_metrics.get("category", {}).get("macro_f1", 0.0)
        else:
            current_eval_metric = overall_macro_f1

        # Mostrar métricas completas calculadas por época
        print(f"\n--- Época {epoch}/{epochs} Completada ---")
        print(f"  Pérdida Entrenamiento (Train Loss): {avg_train_loss:.4f}")
        print(f"  Pérdida Validación (Val Loss):       {val_loss:.4f}")
        print("  Métricas de Validación por Tarea:")
        for t in tasks:
            m = val_metrics.get(t, {})
            print(
                f"    - {t:<14}: Acc={m.get('accuracy', 0.0):.4f} | "
                f"Macro-F1={m.get('macro_f1', 0.0):.4f} | Micro-F1={m.get('micro_f1', 0.0):.4f}"
            )
        print(f"  Métrica de Selección ({metric_target_name}): {current_eval_metric:.4f}")

        # Guardar SIEMPRE el checkpoint de la última época
        save_file(agent.model.state_dict(), last_weights_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "overall_macro_f1": round(overall_macro_f1, 4),
            "val_metrics": val_metrics,
        }
        for t in tasks:
            m = val_metrics.get(t, {})
            epoch_record[f"{t}_macro_f1"] = round(m.get("macro_f1", 0.0), 4)
            epoch_record[f"{t}_accuracy"] = round(m.get("accuracy", 0.0), 4)
            epoch_record[f"{t}_micro_f1"] = round(m.get("micro_f1", 0.0), 4)

        history.append(epoch_record)

        # Comprobar si supera el mejor resultado histórico
        if current_eval_metric > (best_val_metric + 1e-4):
            best_val_metric = current_eval_metric
            best_epoch = epoch
            epochs_without_improvement = 0

            # Guardar el MEJOR modelo
            save_file(agent.model.state_dict(), best_weights_path)
            save_file(agent.model.state_dict(), default_model_path)
            print(f"  -> [MEJOR CHECKPOINT] Guardado en época {epoch} con {metric_target_name}: {best_val_metric:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"  -> [SIN MEJORA] Épocas consecutivas sin mejora: {epochs_without_improvement}/{patience} "
                f"(Mejor época fue {best_epoch} con {best_val_metric:.4f})"
            )

            # Verificar criterio de early stopping
            if epochs_without_improvement >= patience:
                print(
                    f"\n[DETENCIÓN TEMPRANA] Early stopping activado tras {patience} épocas sin mejora. "
                    f"Mejor época: {best_epoch} ({metric_target_name}={best_val_metric:.4f})."
                )
                stopped_early = True
                break

    # Exportar metadatos, configuración del agente, encoder y tokenizer
    agent.cfg["fine_tuned"] = True
    config_export_path = checkpoint_dir / "rl_agent_config.json"
    with config_export_path.open("w", encoding="utf-8") as f:
        json.dump(agent.cfg, f, indent=2, ensure_ascii=False)

    encoder_dir = checkpoint_dir / "encoder"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if hasattr(agent.model, "encoder") and hasattr(agent.model.encoder, "config"):
        agent.model.encoder.config.save_pretrained(str(encoder_dir))
    if hasattr(agent, "tok") and hasattr(agent.tok, "save_pretrained"):
        agent.tok.save_pretrained(str(tokenizer_dir))

    best_sha256 = calculate_file_sha256(best_weights_path) if best_weights_path.exists() else ""
    last_sha256 = calculate_file_sha256(last_weights_path) if last_weights_path.exists() else ""

    # Exportar curvas estructuradas (JSON y CSV)
    curves_json_path = checkpoint_dir / "training_curves.json"
    curves_csv_path = checkpoint_dir / "training_curves.csv"
    if history:
        export_training_curves_json(history, curves_json_path)
        export_training_curves_csv(history, curves_csv_path)

    # Generar gráficos científicos de curvas de entrenamiento
    figures_dir_cfg = cfg.get("output", {}).get("figures_dir", "reports/figures")
    figures_dir = Path(figures_dir_cfg)
    viz_reports: Dict[str, str] = {}
    if history:
        viz_generated = plot_all_training_visualizations(
            history=history,
            output_dir=figures_dir,
            tasks=tasks,
            best_epoch=best_epoch,
        )
        # Réplica en checkpoint_dir/figures para autosuficiencia de artefactos
        plot_all_training_visualizations(
            history=history,
            output_dir=checkpoint_dir / "figures",
            tasks=tasks,
            best_epoch=best_epoch,
        )
        viz_reports = {k: str(v) for k, v in viz_generated.items()}

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round((datetime.now(timezone.utc) - start_time).total_seconds(), 2),
        "device": str(target_device),
        "data_hashes": {
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
        },
        "early_stopping": {
            "patience": patience,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric_name": metric_target_name,
            "best_metric_value": round(best_val_metric, 4),
        },
        "checkpoints": {
            "best": {
                "path": str(best_weights_path),
                "sha256": best_sha256,
                "metric_value": round(best_val_metric, 4),
            },
            "last": {
                "path": str(last_weights_path),
                "sha256": last_sha256,
                "epoch": len(history),
            },
        },
        "visualizations": {
            "curves_json": str(curves_json_path),
            "curves_csv": str(curves_csv_path),
            "figures_dir": str(figures_dir),
            **viz_reports,
        },
        "hyperparameters": {
            "epochs_planned": epochs,
            "epochs_executed": len(history),
            "batch_size": batch_size,
            "lr_encoder": lr_enc,
            "lr_head": lr_head,
            "seed": seed,
            "patience": patience,
        },
        "history": history,
    }

    report_path = checkpoint_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nEntrenamiento finalizado. Artefactos exportados en: {checkpoint_dir}")
    print(f"Mejor Checkpoint (Época {best_epoch}): {best_weights_path} (SHA-256: {best_sha256})")
    print(f"Último Checkpoint: {last_weights_path} (SHA-256: {last_sha256})")
    return report


def main() -> None:
    """Punto de entrada CLI para entrenamiento."""
    parser = argparse.ArgumentParser(description="Fine-tuning de Laya Multilingual")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Ruta a training.yaml")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Dispositivo")
    parser.add_argument("--dry-run", action="store_true", help="Paso de prueba rápido con subconjunto mínimo")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribir épocas")
    parser.add_argument("--patience", type=int, default=None, help="Paciencia de early stopping")
    args = parser.parse_args()

    train_laya(
        config_path=Path(args.config),
        device_name=args.device,
        dry_run=args.dry_run,
        override_epochs=args.epochs,
        override_patience=args.patience,
    )


if __name__ == "__main__":
    main()
