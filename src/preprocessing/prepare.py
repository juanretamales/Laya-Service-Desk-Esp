"""Preparación estructural mínima; NO es pipeline aprobado para entrenamiento."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict
import yaml

REQUIRED = ("ticket_id", "subject", "description", "category", "priority", "support_group")


def normalize(row: Dict[str, Any], mapping: Dict[str, str], source_id: str) -> Dict[str, Any]:
    """Valida y normaliza un registro individual contra el mapeo de columnas."""
    missing = [k for k in REQUIRED if k not in mapping or mapping[k] not in row]
    if missing:
        raise ValueError(f"{source_id}: columnas faltantes: {missing}")
    obj = {k: " ".join(str(row[mapping[k]] or "").split()) for k in REQUIRED}
    if not obj["ticket_id"] or not (obj["subject"] or obj["description"]):
        raise ValueError(f"{source_id}: ticket sin id o contenido")
    obj["source_id"] = source_id
    return obj


def run(config: str, output: str) -> int:
    """Ejecuta la preparación estructural desde el archivo de configuración."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    selected = [s for s in cfg["sources"] if s.get("enabled")]
    if not selected:
        raise ValueError("No hay fuentes habilitadas y verificadas")
    records = []
    for s in selected:
        if s.get("license_verified") is not True:
            raise ValueError(f"Licencia no verificada: {s['id']}")
        with open(s["path"], newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                records.append(normalize(row, s["column_map"], s["id"]))
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"No se sobrescribe: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    """Función de entrada CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Registros exportados: {run(args.config, args.output)}")


if __name__ == "__main__":
    main()
