---
trigger: always_on
---

# Política de Almacenamiento: Archivos Temporales vs Documentación Persistente

Para cualquier archivo temporal, plan de trabajo, borrador, scratchpad o documentación:

1. **Archivos Temporales (`tmp/`):**
   - **Ubicación:** Todo archivo efímero, plan de trabajo iterativo (`tmp/plans/<nombre_plan>.md`), notas de depuración, scripts de prueba desechables o salidas intermedias deben guardarse obligatoriamente dentro de `tmp/`.
   - **Ignorado en Git y Docker:** El contenido de `tmp/` está ignorado (`tmp/*`) en `.gitignore` y `.dockerignore` para no contaminar el historial ni el árbol de trabajo.
   - **Nunca en la raíz ni en `src/`:** Prohibido dejar archivos temporales sueltos en la raíz del proyecto o en las carpetas de código fuente.

2. **Documentación Persistente (`docs/`):**
   - **Ubicación:** Se preserva estrictamente la estructura oficial existente en `docs/` (`docs/01_OBJETIVOS.md`, `docs/02_ARQUITECTURA.md`, ..., `docs/PROGRESS.md`).
   - **Estricta Persistencia:** El directorio `docs/` está reservado exclusivamente para documentación técnica oficial, activa y formal del sistema.
   - **Prohibido contaminar `docs/`:** Bajo ninguna circunstancia se deben guardar planes temporales, borradores o archivos desechables en `docs/`.

3. **No Git commands:**
   - No ejecutar ningún comando de Git.
