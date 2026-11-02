# Índice y Análisis de la Bibliografía Científica (arXiv)

Este documento recopila las referencias oficiales de [arXiv.org](https://arxiv.org) analizadas e incorporadas al marco metodológico y experimental de **Laya Service Desk**.

---

## 1. Documentos Principales de arXiv

### [arXiv:2609.33843](https://arxiv.org/abs/2609.33843)
* **Título:** *Laya as a Typed Probabilistic Assessor: An Independent Reproduction and a Preregistered Study of Calibration and Selective Escalation*
* **Autor:** Gowthamkumar Nandakishore
* **Fecha:** Septiembre de 2026
* **Resumen y Aporte Clave:** Es la referencia académica central sobre Laya como modelo evaluador tipado (`choice`, `score`, `noul`). Define formalmente el cálculo del error de calibración esperado (ECE), la brecha confianza-exactitud con signo (*signed confidence-accuracy gap*) y la formulación de políticas de escalamiento selectivo (*selective escalation cascades*) hacia supervisión humana o segundo nivel.
* **Cita en nuestro estudio:** Utilizado en la sección metodológica para fundamentar el modelo de decisiones tipadas y el análisis de calibración probabilística.

---

### [arXiv:2610.02267](https://arxiv.org/abs/2610.02267)
* **Título:** *Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses*
* **Autor:** Jiawei Li
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Introduce la distinción formal entre **modelos de decisión System-1** (clasificadores discriminativos ultrarrápidos de una sola pasada hacia adelante que devuelven probabilidades) y **modelos System-2** (grandes modelos de lenguaje autorregresivos que generan tokens secuenciales). Demuestra ganancias de varios órdenes de magnitud en costo, latencia y rendimiento para orquestaciones y triaje.
* **Cita en nuestro estudio:** Proporciona la justificación teórica y arquitectónica para adoptar Laya frente a LLMs generativos pesados en centros de soporte técnico.

---

### [arXiv:2610.02586](https://arxiv.org/abs/2610.02586)
* **Título:** *Labels Override Definitions in Jev-Style Typed Decision Models*
* **Autores:** Seyedarmin Azizi, Erfan Baghaei Potraghloo, Massoud Pedram (University of Southern California)
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Investiga el fenómeno por el cual los modelos de decisión tipados atienden predominantemente a las cadenas de texto de las etiquetas (*labels*) por sobre las definiciones o criterios extensos. 
* **Cita en nuestro estudio:** Explica el rendimiento deficiente del baseline Zero-Shot (18.89% en categorías de tickets) debido al desalineamiento léxico intrínseco, justificando plenamente la necesidad de ajuste supervisado (*fine-tuning*) específico de dominio.

---

### [arXiv:2610.03387](https://arxiv.org/abs/2610.03387)
* **Título:** *Benchmarking candidate coverage and rejection policy transfer in typed decision models*
* **Autores:** Jiawen Lu, Tongtong Wu (Monash University)
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Evalúa políticas de abstención y rechazo (*rejection policies*) cuando los conjuntos de candidatos varían o se presentan consultas fuera de distribución (OOD) y omisiones de etiquetas. Compara Laya con Jev y Qwen2.5-7B-Instruct.
* **Cita en nuestro estudio:** Modela el tratamiento de la clase `Other` y la transferencia de políticas de enrutamiento ante incidentes atípicos.

---

### [arXiv:2610.07953](https://arxiv.org/abs/2610.07953)
* **Título:** *Benchmarking System One Models in Online Moderation*
* **Autores:** Federico Mazzoni, Andrea Failla (CNR Italy)
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Analiza modelos System-1 en tareas de moderación y triaje bajo políticas corporativas cambiantes, requiriendo decisiones tipadas auditables y derivación a revisión humana.
* **Cita en nuestro estudio:** Soporte para la gobernanza operativa de decisiones de triaje automatizado en infraestructuras empresariales.

---

### [arXiv:2610.31104](https://arxiv.org/abs/2610.31104)
* **Título:** *DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150*
* **Autores:** Shuang Song, Muhammad Luqman Naseem, Zhongqiao Yuan
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Presenta un framework de clasificación de intenciones orientado a decisiones en entornos de atención al cliente (BANKING77 y CLINC150) utilizando cabezales especialistas de Laya combinados con regularización SAM.
* **Cita en nuestro estudio:** Antecedente directo en clasificación de intenciones de soporte técnico y servicio al usuario.

---

### [arXiv:2610.36311](https://arxiv.org/abs/2610.36311)
* **Título:** *Confidence Is Not a Promise: A Reference Audit of Typed Decision Models*
* **Autor:** Pedro Carvalho Brom (Federal Institute of Brasília)
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Audita de forma rigurosa la interpretación de las distribuciones de probabilidad en modelos de decisión tipados (Laya y Jev 1.13), enfatizando que la confianza debe respaldarse con límites inferiores verificados empíricamente.
* **Cita en nuestro estudio:** Refuerza el protocolo de integridad experimental y el análisis de confiabilidad probabilística.

---

### [arXiv:2610.38697](https://arxiv.org/abs/2610.38697)
* **Título:** *SatOps-Decision: Low-Latency Discriminative Decisions for Onboard Satellite Communication Operations*
* **Autores:** Lifan Mei, Yuxuan Li, Jiaxian Tu, Shenghe Xu (XJTLU, Amazon)
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Primer precedente de adaptación de dominio de un codificador Laya nativo para un entorno operacional crítico en tiempo real (operaciones satelitales). Demuestra una mejora masiva de exactitud (85.3% vs 42.5% frozen) con latencia de 90 ms frente a modelos generativos de más de 400 ms.
* **Cita en nuestro estudio:** Es el paralelo metodológico directo de nuestro trabajo, validando el proceso de ajuste de Laya para triaje de incidentes IT corporativos.

---

### [arXiv:2610.40318](https://arxiv.org/abs/2610.40318)
* **Título:** *Decision and System One models in rheumatology: accuracy and calibration on specialty-training examination questions and diagnostic case vignettes*
* **Autores:** Alfredo Madrid-García, Diego Benavent, Fabian Lechner, Phillip Kremer, et al.
* **Fecha:** Octubre de 2026
* **Resumen y Aporte Clave:** Evalúa modelos System-1 y de decisión en viñetas clínicas diagnósticas en español e inglés (exámenes MIR y RheumBench), validando la capacidad de Laya para decisiones multiclase complejas en idioma español.
* **Cita en nuestro estudio:** Antecedente formal del rendimiento de modelos System-1 en idioma español en tareas de triaje profesional estructurado.

