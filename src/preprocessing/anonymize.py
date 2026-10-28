"""
Módulo de detección, anonimización y sanitización de Información Personal Identificable (PII).

Fase: Preprocesamiento y Calidad de Datos (Fase 2.1 del Roadmap)
Garantías:
    - Enmascaramiento determinista de entidades sensibles: emails, teléfonos, direcciones IP, MACs, tarjetas.
    - Normalización de placeholders preexistentes en fuentes abiertas (ej. [NAME], [EMAIL] -> <NAME>, <EMAIL>).
    - Preservación de la estructura gramatical y semántica para clasificación sin fuga de datos sensibles.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern, Tuple

# DECISIÓN DE ARQUITECTURA: Compilar las expresiones regulares a nivel de módulo para maximizar
# el rendimiento de procesamiento al iterar sobre miles de tickets en entornos CPU o Colab.

# 1. Correos electrónicos estándar
EMAIL_PATTERN: Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# 2. Direcciones IP (IPv4 e IPv6 simplificada)
IPV4_PATTERN: Pattern[str] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
IPV6_PATTERN: Pattern[str] = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
)

# 3. Direcciones MAC
MAC_PATTERN: Pattern[str] = re.compile(
    r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
)

# 4. Números de teléfono (formatos internacionales y locales de España/Latinoamérica)
PHONE_PATTERN: Pattern[str] = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)

# 5. Tarjetas de crédito / identificadores numéricos de 16 dígitos
CARD_PATTERN: Pattern[str] = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

# 6. URLs y enlaces web
URL_PATTERN: Pattern[str] = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)"
)

# 7. Documentos de identidad españoles y comunes (DNI/NIE con letra de control)
DNI_NIE_PATTERN: Pattern[str] = re.compile(
    r"\b(?:[XYZxyz]?\d{7,8}[A-Za-z])\b"
)

# 8. Mapeo para estandarizar placeholders heredados (ej. Zenodo o Kaggle: [NAME] -> <NAME>)
LEGACY_TAGS_MAP: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"\[NAME\]", re.IGNORECASE), "<NAME>"),
    (re.compile(r"\[COMPANY\]", re.IGNORECASE), "<ORG>"),
    (re.compile(r"\[LOCATION\]", re.IGNORECASE), "<LOC>"),
    (re.compile(r"\[SERVER\]", re.IGNORECASE), "<SERVER>"),
    (re.compile(r"\[TICKET\s*ID\]", re.IGNORECASE), "<TICKET_ID>"),
    (re.compile(r"\[MAC\s*ADDRESS\]", re.IGNORECASE), "<MAC>"),
    (re.compile(r"\[EMAIL\]", re.IGNORECASE), "<EMAIL>"),
    (re.compile(r"\[PHONE\]", re.IGNORECASE), "<PHONE>"),
]


def anonymize_text(text: str) -> str:
    """
    Sanitiza y enmascara todas las entidades sensibles identificadas en una cadena de texto.

    Args:
        text: Texto crudo del ticket o asunto a procesar.

    Returns:
        Cadena de texto con entidades PII reemplazadas por tokens canónicos (<EMAIL>, <PHONE>, etc.).
    """
    if not text:
        return ""

    sanitized = str(text)

    # 1. Normalizar etiquetas preexistentes si la fuente ya venía parcialmente scrubbed
    for pattern, replacement in LEGACY_TAGS_MAP:
        sanitized = pattern.sub(replacement, sanitized)

    # 2. Aplicar filtros de alta especificidad primero (URLs antes que IPs o emails)
    sanitized = URL_PATTERN.sub("<URL>", sanitized)
    sanitized = EMAIL_PATTERN.sub("<EMAIL>", sanitized)
    sanitized = IPV4_PATTERN.sub("<IP>", sanitized)
    sanitized = IPV6_PATTERN.sub("<IP>", sanitized)
    sanitized = MAC_PATTERN.sub("<MAC>", sanitized)
    sanitized = CARD_PATTERN.sub("<CARD>", sanitized)
    sanitized = DNI_NIE_PATTERN.sub("<ID_DOC>", sanitized)
    sanitized = PHONE_PATTERN.sub("<PHONE>", sanitized)

    # 3. Limpiar espacios redundantes generados por reemplazos
    sanitized = " ".join(sanitized.split())

    return sanitized


def detect_pii(text: str) -> Dict[str, int]:
    """
    Detecta y cuantifica la cantidad de entidades sensibles encontradas en un texto.

    Args:
        text: Texto a auditar.

    Returns:
        Diccionario con el recuento de coincidencias por tipo de entidad.
    """
    if not text:
        return {}

    counts: Dict[str, int] = {
        "email": len(EMAIL_PATTERN.findall(text)),
        "ipv4": len(IPV4_PATTERN.findall(text)),
        "ipv6": len(IPV6_PATTERN.findall(text)),
        "mac": len(MAC_PATTERN.findall(text)),
        "card": len(CARD_PATTERN.findall(text)),
        "phone": len(PHONE_PATTERN.findall(text)),
        "url": len(URL_PATTERN.findall(text)),
        "id_doc": len(DNI_NIE_PATTERN.findall(text)),
    }
    return {k: v for k, v in counts.items() if v > 0}


def anonymize_record(
    record: Dict[str, Any],
    text_fields: Tuple[str, ...] = ("subject", "description"),
) -> Dict[str, Any]:
    """
    Aplica la sanitización PII sobre los campos textuales de un diccionario de registro.

    Args:
        record: Diccionario representativo de un ticket de soporte.
        text_fields: Tupla con los nombres de las claves textuales a anonimizar.

    Returns:
        Nuevo diccionario con los campos especificados anonimizados.
    """
    sanitized_record = dict(record)
    for field in text_fields:
        if field in sanitized_record and sanitized_record[field] is not None:
            sanitized_record[field] = anonymize_text(str(sanitized_record[field]))
    return sanitized_record
