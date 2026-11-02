# Carpeta de Publicación y Divulgación (`paper/`)

Esta carpeta está dedicada a la confección, redacción y materiales de publicación de los resultados de **Laya Service Desk**, manteniendo el resto del proyecto limpio y aislado.

## Documentos Disponibles

### 1. Artículo para Sitio Web (Recomendado para Divulgación)
* **[`ARTICULO_WEB_LAYA_FINETUNING_ESPANOL.md`](ARTICULO_WEB_LAYA_FINETUNING_ESPANOL.md):** 
  Artículo técnico ágil y directo pensado para publicación en blog personal, LinkedIn o web corporativa.
  * **Premisa central:** Por qué hacer fine-tuning a Laya en español es mucho más fácil, económico y efectivo que usar un LLM gigante.
  * **Puntos fuertes:** Comparativa System-1 vs. System-2, costo menor a \$1 USD, salto de exactitud de 18.89% a 83.56% ($4.42\times$), cero alucinación y despliegue local on-premise con compuertas de confianza (*Human-in-the-Loop*).

### 2. Manuscrito y Fuentes Académicas (Para Referencia o Archivo)
* **[`MANUSCRIPT.md`](MANUSCRIPT.md):** Versión académica completa en Markdown con fórmulas matemáticas KaTeX y tablas exhaustivas.
* **[`main.tex`](main.tex):** Manuscrito formal en LaTeX (IEEEtran / arXiv).
* **[`references.bib`](references.bib):** Bibliografía consolidada en formato BibTeX con fuentes verificadas de arXiv.org.
* **[`BIBLIOGRAPHY_INDEX.md`](BIBLIOGRAPHY_INDEX.md):** Índice analítico detallado de los 9 artículos científicos reales recopilados de arXiv.
* **`figures/`:** Figuras vectoriales y gráficos a 300 DPI (`training_dashboard.png`, etc.).

## Compilación del Documento LaTeX

El archivo `main.tex` es totalmente autocontenido. Puede compilarse con:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
O cargarse directamente en una carpeta o proyecto de [Overleaf](https://www.overleaf.com/).

