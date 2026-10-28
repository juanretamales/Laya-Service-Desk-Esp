# Sincronización GitHub privado ↔ equipo ↔ Colab Pro

## Principios
GitHub es la fuente de verdad **solo para código y metadatos**. Colab es entorno de ejecución efímero, no editor principal. Google Drive guarda datasets y checkpoints. El repositorio privado **no sustituye** anonimización, controles de acceso ni revisión de licencias.

## Operaciones de solo lectura (cuando se autoricen comandos Git)
- Consultar estado y diferencias del repositorio.
- Revisar revisión exacta usada por Colab y el manifiesto del dataset.
- En Colab, obtener una revisión fijada del código por medio de un mecanismo de autenticación aprobado; nunca pegar tokens en notebooks ni URLs.

## Operaciones que escriben (requieren autorización explícita)
- Crear repositorio privado, inicializar Git, añadir remoto, hacer commits y enviar cambios.
- Actualizar ramas o sincronizar cambios entre local y remoto.
- Guardar o sobrescribir checkpoints y resultados en Drive.

## Flujo propuesto (no ejecutado)
1. Desarrollar y probar en Docker local.
2. Revisar diffs, datos sensibles y licencias.
3. Solicitar autorización para registrar/publicar cambios en GitHub.
4. En Colab cargar código desde revisión inmutable autorizada; registrar hash de revisión, entorno y hashes SHA-256 de datasets.
5. Montar Drive tras autorización y guardar resultados en carpeta única por `run_id`, sin sobrescribir.
6. Revisar resultados localmente; incorporar solo métricas agregadas no sensibles al repositorio con autorización.

## Conflictos y secretos
No trabajar simultáneamente sobre el mismo archivo desde Colab y Antigravity. No hacer `push` desde Colab por defecto. Guardar credenciales en gestores de secretos o mecanismos oficiales de autenticación, nunca en `.env` versionado ni en notebooks. Evitar tokens en logs. Si se publica el repositorio posteriormente, auditar historial y licencias.

## Registro mínimo
`run_id`, fecha UTC, revision del codigo, version y hash del dataset, identificador y revision del modelo, semilla, hardware, entorno, configuracion, métricas, costo estimado y rutas de artefactos. Nunca registrar tickets con datos personales en reportes públicos.
