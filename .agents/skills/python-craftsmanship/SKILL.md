---
name: python-craftsmanship
description: >-
  Use this skill whenever adding, refactoring, or modifying Python code in Laya Service Desk.
  It enforces rigorous code craftsmanship: PEP 8 style (line length <= 120, flake8 = 0 errors),
  PEP 20 Zen of Python design principles, PEP 560/484 strict type hinting, comprehensive docstrings
  (with explicit domain purpose and pipeline stage differentiation), experimental rationale comments,
  and layer separation across ingestion, preprocessing, training, evaluation, and inference.
---

# Skill: Artesanía y Calidad de Código Python (`python-craftsmanship`)

## 1. Propósito
Asegurar que cada línea de código Python escrita o modificada en **Laya Service Desk** cumpla con los más altos estándares de calidad, legibilidad, robustez, reproducibilidad y documentación, preservando la arquitectura modular del laboratorio experimental y garantizando que los pipelines de ML sean trazables, seguros y verificables.

---

## 2. Cuándo Activar Esta Skill
- Al crear o modificar cualquier archivo `.py` en:
  - `src/ingestion/` (ingesta de fuentes y datasets)
  - `src/preprocessing/` (normalización, anonimización, deduplicación, validación)
  - `src/training/` (entrenamiento, fine-tuning, gestión de pesos e hiperparámetros)
  - `src/evaluation/` (benchmark, métricas F1/Accuracy, matriz de confusión, costo computacional)
  - `src/inference/` (carga de modelo, inferencia, cuantización post-training INT8/INT4)
  - `tests/` (suites de prueba unitarias y de integración)
  - Scripts en `notebooks/` o utilitarios auxiliares.
- Al refactorizar lógica existente o corregir bugs.

---

## 3. Los 5 Pilares de Calidad

### Pilar 1: PEP 8 (Estilo, Formato y Limpieza)
1. **Longitud de Línea:** Máximo **120 caracteres** por línea. Romper expresiones largas, listas y cadenas formateadas con sangría colgante limpia.
2. **Organización de Importaciones:**
   - **Grupo 1:** Librería estándar de Python (`argparse`, `csv`, `json`, `pathlib`, `typing`, `logging`, `os`, `sys`).
   - **Grupo 2:** Dependencias de terceros (`yaml`, `pytest`, `torch`, `transformers`, `datasets`, `sklearn`).
   - **Grupo 3:** Módulos internos del proyecto (`src.preprocessing.*`, `src.evaluation.*`, etc.).
   - **Regla:** Cero importaciones no utilizadas (`F401`).
3. **Espaciado y Formato:**
   - 2 líneas en blanco entre funciones y clases de nivel superior.
   - 1 línea en blanco entre métodos dentro de una clase.
   - No compactar sentencias con punto y coma (`;`).
4. **Verificación Automatizada:**
   ```bash
   python -m flake8 src/ tests/ --max-line-length=120
   ```
   *Debe arrojar 0 errores.*

---

### Pilar 2: PEP 20 (El Zen de Python & Arquitectura Experimental)
- **Explicit is better than implicit:**
  - Esquemas de datos claros. Las columnas obligatorias, categorías, prioridades y grupos de soporte deben definirse explícitamente en constantes o esquemas.
  - Sanitizar y validar tipos explícitamente (`str(val or '').strip()`).
- **Separación Estricta de Capas:**
  - **`src/preprocessing`:** Solo transforma datos crudos a formatos normalizados (JSONL/datasets). No debe contener dependencias de entrenamiento pesado ni modelos.
  - **`src/training`:** Solo consume datos procesados y validados. No debe realizar transformaciones de datos no documentadas.
  - **`src/evaluation`:** Mide métricas objetivas (F1-macro, exactitud, latencia, VRAM) sobre conjuntos de prueba inmutables. Nunca altera los pesos ni altera el split de test.
  - **`src/inference`:** Expone la predicción y cuantización sin depender de la lógica de entrenamiento.
- **Errors should never pass silently:**
  - Prohibido silenciar errores con `except: pass`.
  - Fallar de manera temprana y explícita ante datasets sin licencia verificada, columnas faltantes o datos sin anonimizar.
  - Placeholders (`train.py`, `benchmark.py`) deben lanzar `NotImplementedError` explícito hasta que los gates del protocolo se cumplan.

---

### Pilar 3: PEP 560 / PEP 484 (Anotaciones de Tipos - Typing)
Todas las funciones, métodos y clases deben contar con tipado estricto en parámetros y retorno:
- **Tipos Estándar:** `int`, `float`, `str`, `bool`, `Path`, `Optional[str]`.
- **Colecciones:** `List[Dict[str, Any]]`, `Dict[str, str]`, `Tuple[str, ...]`, `Set[str]`.
- **Estructuras de ML:** `Dict[str, float]` para métricas, `Dict[str, Any]` para configuraciones.
- **Retornos vacíos:** Retorno explícito `-> None` si la función no retorna valor.

#### Ejemplo Estándar en Laya Service Desk:
```python
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "ticket_id",
    "subject",
    "description",
    "category",
    "priority",
    "support_group",
)


def normalize_record(
    row: Dict[str, Any],
    column_mapping: Dict[str, str],
    source_id: str,
) -> Dict[str, str]:
    """
    Normaliza un registro individual de ticket a partir del mapeo de columnas provisto.

    Args:
        row: Diccionario con los datos crudos de la fila.
        column_mapping: Mapeo de columnas requeridas hacia los nombres en la fuente.
        source_id: Identificador de la fuente para trazabilidad de errores.

    Returns:
        Diccionario con las columnas requeridas normalizadas y la fuente asociada.

    Raises:
        ValueError: Si faltan columnas requeridas o el ticket carece de contenido.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in column_mapping or column_mapping[col] not in row]
    if missing:
        raise ValueError(f"[{source_id}] Columnas faltantes en registro: {missing}")

    normalized: Dict[str, str] = {
        col: " ".join(str(row[column_mapping[col]] or "").split())
        for col in REQUIRED_COLUMNS
    }

    if not normalized["ticket_id"] or not (normalized["subject"] or normalized["description"]):
        raise ValueError(f"[{source_id}] Registro inválido: ticket sin ID o sin contenido descriptivo")

    normalized["source_id"] = source_id
    return normalized
```

---

### Pilar 4: Documentación Obligatoria y Docstrings

#### 1. Docstrings de Módulo
Al inicio de cada archivo `.py`, incluir un docstring que explique su propósito y fase en el pipeline:
```python
"""
Módulo de normalización y limpieza de tickets para Laya Service Desk.

Fase: Preprocesamiento (CSV -> JSONL normalizado)
Garantías: Sanitización de texto, verificación de columnas mínimas y trazabilidad de fuente.
"""
```

#### 2. Docstrings de Funciones y Clases
- Modo imperativo en la primera línea.
- Secciones `Args:`, `Returns:` y `Raises:` cuando aplique.

---

### Pilar 5: Comentarios de Decisiones Científicas y de Arquitectura (*Rationale*)
Documentar siempre el **por qué** detrás de decisiones experimentales críticas, filtros de datos o selección de hiperparámetros.

**Formato recomendado:**
```python
# DECISIÓN EXPERIMENTAL: Se aísla el split de test antes de cualquier ajuste y se mantiene
# inmutable para evitar data leakage entre fine-tuning y evaluación de baseline.
```
```python
# DECISIÓN DE ARQUITECTURA: La inferencia en desarrollo se limita a CPU para garantizar
# que los tests y validaciones se puedan ejecutar localmente sin depender de GPU/CUDA.
```

---

## 4. Checklist de Verificación Antes de Finalizar Cada Tarea

- [ ] ¿El código cuenta con tipado estricto en parámetros y retornos (`PEP 484`)?
- [ ] ¿Los módulos y funciones cuentan con docstrings explicativos (`PEP 257`)?
- [ ] ¿Se añadieron comentarios `# DECISIÓN EXPERIMENTAL:` o `# DECISIÓN DE ARQUITECTURA:` donde aplique?
- [ ] ¿Se ejecutó `python -m flake8 src/ tests/ --max-line-length=120` con **0 errores**?
- [ ] ¿Se ejecutó la suite de tests (`python -m pytest -q`) y pasan todos los casos?
- [ ] ¿Se verificó que no se incluyan credenciales, PII sin anonimizar, ni rutas absolutas?
- [ ] **¿Los archivos temporales/scratchpads se ubicaron en `tmp/` y la documentación persistente en `docs/`?**
- [ ] **¿Se respetó la regla de no ejecutar comandos Git?**
- [ ] **¿Todas las referencias de archivos usan rutas relativas dentro del workspace?**
