# Laya Service Desk — laboratorio reproducible

**Estado:** arquitectura y contratos iniciales; sin descarga de datos, pesos, GPU, entrenamiento ni repositorios remotos.

## Objetivo
Estimar el costo computacional y la mejora de calidad del **mismo Laya Multilingual** antes y después de fine-tuning en clasificación de tickets corporativos en español. Como experimento final, cuantizar el modelo ajustado y evaluar el compromiso calidad/eficiencia. Clasificar **categoría, prioridad y grupo de soporte** por separado, solo donde existan etiquetas verificadas.

## Arquitectura
- Antigravity IDE en Windows: editor; abrir carpeta mediante Dev Containers si la extensión es compatible.
- Docker CPU-only: Python, scripts y tests; no GPU ni CUDA local.
- GitHub privado: fuente de verdad para código/configuración/documentación, sin datos sensibles ni pesos.
- Colab Pro: ejecutar notebooks con GPU remota, sin prometer disponibilidad ni hardware específico.
- Google Drive: datasets y artefactos experimentales grandes, con control de acceso y manifiestos.

## Empezar
1. Leer `docs/10_DOCKER_ANTIGRAVITY.md`, `docs/11_GITHUB_COLAB_SYNC.md` y `docs/12_PROTOCOLO_EXPERIMENTAL.md`.
2. Revisar el impacto y solicitar autorización antes de construir/arrancar contenedores o instalar dependencias.
3. Abrir carpeta en contenedor con `.devcontainer/devcontainer.json` (si compatible), o usar `docker compose run --rm dev` tras autorizar.
4. Ejecutar pruebas CPU: `python -m pytest -q` dentro del contenedor.
5. Auditar fuentes/licencias, anonimización, etiquetas y particiones antes de cualquier entrenamiento.
6. El notebook de Colab es un contrato/documento, no descarga ni entrena automáticamente.

**Limitaciones:** la API y licencia de Laya, pesos, métodos de entrenamiento y cuantización requieren verificación previa; no asumir compatibilidad de Hugging Face ni métricas. Los módulos `train.py` y `benchmark.py` son placeholders que fallan explícitamente.
