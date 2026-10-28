# Normalización y control de calidad

Contrato de entrada CSV UTF-8 (por fuente): `ticket_id,subject,description,category,priority,support_group`. Los nombres de columnas reales deben mapearse explícitamente en YAML. Si faltan etiquetas, no inventarlas: marcar `unknown` y documentar que el ejemplo no sirve para supervisar ese campo.

Etiquetas canónicas iniciales (propuestas): categorías `access,network,hardware,software,email,database,other,unknown`; prioridades `low,medium,high,critical,unknown`; grupos `service_desk,identity,network,workplace,applications,dba,other,unknown`. Revisar con expertos y versionar mapeos. No mapear automáticamente por similitud semántica sin revisión.

El preparador actual valida estructura, normaliza espacios y exporta JSONL. **No** detecta idiomas, anonimiza ni deduplica: estas tareas son gates obligatorios antes de entrenamiento. No usar salida directamente para entrenar hasta completar los gates.
