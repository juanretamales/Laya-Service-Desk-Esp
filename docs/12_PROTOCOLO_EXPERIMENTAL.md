# Protocolo científico y paper

## Preguntas
1. ¿Cuánto tiempo/GPU-horas y costo monetario estimado demanda el fine-tuning?
2. ¿Cuánto cambia F1-macro en español respecto del mismo Laya original?
3. ¿La cuantización posterior mejora eficiencia (memoria, tamaño, latencia) sin pérdida relevante de calidad?

## Fases y gates
0. **Verificar Laya:** repositorio oficial, pesos, licencia, API de clasificación, disponibilidad de inferencia sin entrenamiento, fine-tuning y cuantización. Bloquear experimentos si no se valida.
1. **Datasets:** verificar fuentes, licencias, español, etiquetas, anonimización, mapeo por tarea, duplicados y fuga entre particiones. Fijar test inmutable; separar por fuente/grupo si corresponde.
2. **Baseline:** evaluar Laya original con el mismo esquema de categorías, prompts/entrada y test, sin entrenar. Si requiere cabezal aprendido, declarar que no existe baseline directamente comparable y diseñar control válido antes de continuar.
3. **Fine-tuning:** entrenar solo con train, seleccionar hiperparámetros con validation, medir tiempo total, GPU-horas, VRAM pico, GPU asignada, semillas, costo según tarifa verificable. Prohibido usar test para decisiones de entrenamiento.
4. **Comparación:** ejecutar test idéntico, reportar F1-macro, accuracy, F1 por clase, matriz de confusión y dispersión entre semillas cuando sea viable.
5. **Cuantización post-training:** intentar INT8 y opcional INT4 **solo si compatibles**, evaluar mismo test, tamaño, RAM/VRAM, latencia, costo de conversión y variación F1.
6. **Paper:** metodología, procedencia/licencias, limitaciones, sesgos de idioma, recursos, costo observado vs estimado, intervalos, resultados negativos y reproducibilidad.

## Reglas
No afirmar mejoras sin evidencia; no fabricar etiquetas. Prioridad y grupo de soporte pueden no ser inferibles del texto: evaluar disponibilidad y consistencia. Colab Pro no garantiza GPU específica. Mantener el modelo original y el ajustado en precisión comparable para aislar fine-tuning.
