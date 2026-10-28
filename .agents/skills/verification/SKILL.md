---
name: verification
description: >-
  Mandatory gate before claiming any task is complete in Laya Service Desk. Requires running actual
  verification commands in the terminal and showing fresh evidence in the output — zero "should work" claims without proof.
---

# Skill: Verificación Obligatoria con Evidencia (`verification`)

## 1. Regla de Oro
> [!CAUTION]
> **Queda estrictamente prohibido afirmar que una tarea está "lista", "resuelta", que "debería funcionar" o que "el código se ve correcto" sin haber ejecutado los comandos reales de verificación en la terminal y presentado la evidencia exacta de la salida.**

---

## 2. Matriz de Verificación para Laya Service Desk

### A. Preprocesamiento y Normalización de Datos (`src/preprocessing/`, `src/ingestion/`)
Antes de dar por completado un cambio en scripts o funciones de preparación de datos:
1. **Tests Unitarios de Normalización:**
   ```bash
   python -m pytest tests/test_prepare.py -v
   ```
   *Criterio:* Todos los tests deben reportar `PASSED`.
2. **Validación de Salvaguardas y Formato:**
   - Confirmar que se prevenga la sobreescritura accidental de datasets existentes.
   - Confirmar que se validen los campos mínimos obligatorios (`ticket_id`, `subject`, `description`, `category`, `priority`, `support_group`).
   - Confirmar que las fuentes no verificadas (`license_verified != True`) sean rechazadas explícitamente.

---

### B. Configuraciones y Manifiestos (`configs/`)
Al crear o modificar archivos de configuración YAML:
1. **Validación de Sintaxis YAML:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/<archivo.yaml>', encoding='utf-8'))"
   ```
   *Criterio:* Cero errores de parseo.
2. **Chequeo de Esquema de Columnas:**
   - Verificar que todos los `column_map` mapeen contra las columnas requeridas sin campos nulos.

---

### C. Módulos de Entrenamiento, Inferencia y Benchmark (`src/training/`, `src/evaluation/`, `src/inference/`)
1. **Respeto a los Gates Experimentales (G0-G6):**
   - Prohibido simular métricas, fabricar salidas o inventar baselines.
   - Si una fase aún no cuenta con modelo verificado, pesos aprobados o datos congelados, el módulo debe mantener su `NotImplementedError` explícito.
2. **Compatibilidad CPU / Entorno Local:**
   - Verificar que el código de desarrollo y pruebas no intente reservar memoria CUDA o GPU si se ejecuta en el contenedor CPU-only.

---

### D. Verificación Transversal de Calidad (`python-craftsmanship`)
1. **Sintaxis y Compilación:**
   ```bash
   python -m py_compile <ruta/al/archivo_modificado.py>
   ```
2. **Estilo y Linting (PEP 8):**
   ```bash
   python -m flake8 src/ tests/ --max-line-length=120
   ```
   *Criterio:* **0 errores**.
3. **Tipado Estricto (PEP 484/560):**
   - Todos los argumentos y retornos con type hints válidos.
4. **Documentación:**
   - Docstrings presentes en módulos, funciones y clases explicando fase y rol en el pipeline.

---

### E. Suite de Pruebas General
Ejecutar la suite completa antes de cerrar cualquier tarea:
```bash
python -m pytest -q
```
*(O mediante Docker si se trabaja dentro del contenedor: `docker compose run --rm dev`)*.

---

### F. Actualización del Grafo de Código (Graphify)
Si se modificaron o crearon archivos de código `.py` y el directorio `graphify-out/` existe:
```bash
graphify update .
```

---

## 3. Formato Obligatorio de Reporte de Evidencia

Al responder al usuario para cerrar una tarea o fase, incluye siempre el bloque de evidencia:

```markdown
### 📋 Evidencia de Verificación

#### 1. Comandos Ejecutados
* `python -m pytest tests/ -v` — Todos los tests aprobados.
* `python -m flake8 src/ tests/ --max-line-length=120` — 0 errores de estilo.

#### 2. Salida de Terminal
```text
tests/test_prepare.py::test_normalize PASSED [ 33%]
tests/test_prepare.py::test_missing_column PASSED [ 66%]
tests/test_prepare.py::test_disabled_sources PASSED [100%]
============================== 3 passed in 0.05s ==============================
```
```

---

## 4. Qué Cuenta como Evidencia vs Qué NO Cuenta

| ✅ SÍ Cuenta como Evidencia | ❌ NO Cuenta como Evidencia (Inválido) |
| :--- | :--- |
| Salida real de `pytest` mostrando `PASSED`. | *"Revisé el código y la lógica se ve bien."* |
| Salida de `flake8` con 0 errores. | *"Seguí el mismo patrón que el otro script, así que funciona."* |
| Salida de compilación limpia con `python -m py_compile`. | *"Debería andar sin problemas."* |
| Salida de validación de YAML sin errores. | *"No debería haber errores de configuración."* |
| Ejecución de dry-run real con salida en consola. | Afirmar que está listo sin haber corrido ningún comando. |
