# Fuentes de Datos, Auditoría de Licencias y Matriz de Riesgos (Fase 0)

Este documento registra los resultados formales de la auditoría de licencias, disponibilidad de datos y análisis de riesgos para los datasets candidatos evaluados en la **Fase 0** (Gates G0.3 y G0.4).

---

## 📊 1. Matriz de Auditoría de Fuentes Candidatas

| ID | Nombre de Fuente | URL / Identificador | Licencia Verificada | Idioma Español | Datos Descargables | Veredicto Formal |
|---|---|---|:---:|:---:|:---:|:---:|
| **`zenodo_tickets`** | Classification of IT Support Tickets | [DOI: 10.5281/zenodo.7648117](https://zenodo.org/records/7648117) | **CC BY 4.0** | Sí (multilingüe EN/DE/PT/ES) | Sí (`X_train.csv`, `y_train.csv`, etc.) | **APROBADO** |
| **`kaggle_multilingual`** | Customer IT Support - Ticket Dataset (T. Bück) | [Kaggle / Hugging Face](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) | **CC-BY-NC 4.0** | Sí (columna `Language == 'es'`) | Sí (CSV estructurado con `Queue`, `Priority`, `Body`) | **APROBADO (Académico/No Comercial)** |
| **`mendeley_helpdesk`** | Help Desk Tickets (M. Abdellatif) | [DOI: 10.17632/btm76zndnt.3](https://data.mendeley.com/datasets/btm76zndnt/3) | **CC BY 4.0** | Limitado (diálogos en inglés) | Sí (`issues.csv`, `sample_utterances.csv`) | **CONDICIONAL / SECUNDARIO** |
| **`mdpi_it_incidents`** | Classifying IT Incident Tickets (Oliveira et al.) | [DOI: 10.3390/ai3030035](https://www.mdpi.com/2673-2688/3/3/35) | N/A (Paper) | No | **No** (*"Not applicable. Data in manuscript tables"*) | **EXCLUIDO FORMALMENTE** |

---

## 🔍 2. Detalle de Fuentes Evaluadas

### A. `zenodo_tickets` (Aprobado Principal)
* **Licencia:** `Creative Commons Attribution 4.0 International (CC BY 4.0)`.
* **Uso permitido:** Compartir, adaptar y redistribuir con fines académicos y comerciales dando atribución.
* **Características:**
  * 2.229 tickets clasificados manualmente por 3 profesionales de soporte IT (Florianópolis, Brasil).
  * 7 categorías canónicas de soporte de infraestructura y sistemas.
  * Multi-idioma con presencia verificada de español.
* **Privacidad (PII):** Pre-sanitizado en 3 etapas (AWS Comprehend PII + Regex + verificación manual). Reemplazo por etiquetas `[NAME]`, `[EMAIL]`, `[MAC ADDRESS]`.

### B. `kaggle_multilingual` (Aprobado No Comercial)
* **Licencia:** `Creative Commons Attribution-NonCommercial 4.0 (CC-BY-NC 4.0)`.
* **Uso permitido:** Investigación científica, benchmarking y publicación no comercial.
* **Características:**
  * Mapeo directo a las 3 tareas del proyecto:
    * `Queue` $\rightarrow$ `support_group` (Technical Support, Billing, Customer Service, etc.).
    * `Priority` $\rightarrow$ `priority` (Low, Medium, Critical).
    * `Type` / `Tags` $\rightarrow$ `category` (Incident, Request, Problem, Software Bug, etc.).
    * `Subject` $\rightarrow$ `subject`.
    * `Body` $\rightarrow$ `description`.
  * Columna explícita `Language`: permite aislar los registros en español de forma determinista.

### C. `mendeley_helpdesk` (Secundario / Reserva)
* **Licencia:** `CC BY 4.0`.
* **Limitación:** El archivo principal `issues.csv` contiene metadatos de tickets y tiempos de resolución, pero el texto descriptivo está restringido a `sample_utterances.csv` en inglés. Se mantiene como fuente secundaria de respaldo.

### D. `mdpi_it_incidents` (Excluido)
* **Causa de exclusión:** La declaración oficial del artículo indica *"Data Availability Statement: Not applicable in this study"*. Solo incluye tablas agregadas de resultados dentro del manuscrito; no existe archivo crudo descargable. Queda descartado en cumplimiento de la regla de admisión de datos.

---

## ⚠️ 3. Matriz de Riesgos y Mitigaciones (Fase 0.4)

| ID | Factor de Riesgo | Nivel | Impacto Potencial | Estrategia de Mitigación |
|:---:|---|:---:|---|---|
| **R1** | **Licencia Restrictiva (`CC-BY-NC`)** | Medio | Impide comercializar los pesos derivados directamente de Kaggle. | Mantener los experimentos bajo alcance estrictamente académico/paper de investigación. Aislar fuentes en el manifiesto para trazabilidad. |
| **R2** | **PII Residual o Incompleta** | Alto | Violación de privacidad o fuga de datos personales en checkpoints públicos. | Implementar capa obligatoria de Regex (`src/preprocessing/anonymize.py`) sobre todos los campos de texto antes de generar el split final. |
| **R3** | **Desbalance de Clases y Escasez en Español** | Medio | Rendimiento deficiente en clases minoritarias de `support_group` o `priority`. | Emplear métricas macro-ponderadas (F1-macro), estratificación rigurosa en el split (70/15/15) y marcar clases faltantes como `unknown`. |
| **R4** | **Límites de Longitud de Contexto (Tokens)** | Bajo | Truncamiento de tickets muy extensos durante inferencia o fine-tuning. | `laya-multilingual` soporta hasta 8.192 tokens (`max_len=8192`). Analizar la distribución de longitud de caracteres en Fase 2 para definir el `max_len` óptimo (1.024 cubre >98% de tickets). |
| **R5** | **Recursos y Desconexión en Colab Pro** | Medio | Pérdida de estado o consumo excesivo de unidades de cómputo GPU. | Checkpointing periódico por época hacia Google Drive; scripts ejecutables de forma no interactiva con `seed` fijado. |
