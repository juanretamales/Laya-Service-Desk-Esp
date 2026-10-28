# Roadmap para agente IA — Laya Service Desk

La IA puede subdividir fases y registrar progreso en `docs/PROGRESS.md`; cada fase exige pruebas y aprobación previa de escrituras, descargas, GPU y operaciones externas. No usar Git sin autorización expresa.

- Fase 0: inspección **solo lectura** de fuentes, licencias y API/pesos de Laya; matriz de riesgos.
- Fase 1: adaptadores de datasets y fixtures sintéticos, previo permiso para descargar.
- Fase 2: PII, normalización español, etiquetas, deduplicación y splits sin fuga.
- Fase 3: baseline original con métricas verificadas.
- Fase 4: fine-tuning y registro de costos en Colab.
- Fase 5: comparación estricta con test congelado.
- Fase 6: cuantización posterior y reevaluación.
- Fase 7: paper, evidencias y auditoría de reproducibilidad.

Antes de cada tarea: distinguir **solo lectura**, **reversible** y **difícil de revertir**; explicar impacto, destino, rollback y pedir autorización para cambios. No inventar compatibilidad, resultados ni licencias.
