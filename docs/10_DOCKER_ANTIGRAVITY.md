# Docker + Antigravity (CPU-only)

## Solo lectura
Inspeccionar `Dockerfile`, `compose.yaml`, `.devcontainer/devcontainer.json`, rutas montadas y versiones. No requiere drivers NVIDIA ni CUDA local para este flujo.

## Cambios reversibles sujetos a autorización
Construir la imagen descarga dependencias e instala paquetes **dentro de la imagen**; usa disco y red. Ejecutar el servicio monta el proyecto en modo lectura/escritura y las pruebas podrían escribir cachés. Antes de hacerlo, confirmar que no haya archivos productivos en la carpeta. Reversión: detener contenedor, eliminar imagen/caché creada y conservar fuentes; no usar `docker system prune`.

## Apertura
Antigravity puede intentar Dev Containers si dispone de una extensión compatible; **compatibilidad no verificada**. Alternativa: editar en Antigravity y ejecutar comandos dentro de Docker. No montar `/var/run/docker.sock`, no activar `privileged` ni configurar GPU local.

## Permisos
Contenedor con UID 10001; en bind mounts Linux/WSL puede haber discrepancias de UID/GID. Revisar propietario de la carpeta antes de iniciar; no aplicar `chmod -R` ni `chown -R` automáticamente. Windows con Docker Desktop requiere instalación/WSL2 previamente disponibles.
