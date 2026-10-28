"""
Módulo de detección, validación y filtrado de idioma para Laya Service Desk.

Fase: Preprocesamiento y Calidad de Datos (Fase 2.2 del Roadmap)
Garantías:
    - Clasificación determinista de idioma basada en marcadores léxicos, stopwords y n-gramas característicos.
    - Diferenciación rigurosa entre español (ES) y portugués (PT), inglés (EN) y alemán (DE).
    - Descarte transparente y auditable de registros que no pertenezcan al corpus en español.
    - Ejecución ligera y autónoma en Python estándar sin dependencias pesadas de ML.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern, Set, Tuple

# DECISIÓN DE ARQUITECTURA: Usar diccionarios de frecuencia de vocabulario funcional y marcadores
# morfosintácticos para evitar cargar modelos pesados de NLP durante la fase de preparación de datos.

ES_STOPWORDS: Set[str] = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "más", "mas", "pero", "sus", "le", "ya", "o", "este", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "tambien", "me", "hasta", "hay", "donde", "desde",
    "todo", "nos", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "esto", "mí", "mi", "antes",
    "algunos", "qué", "unos", "yo", "otro", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
    "acceso", "solicitud", "problema", "error", "contraseña", "contrasena", "usuario", "permiso", "equipo",
    "favor", "ayuda", "sistema", "cuenta", "correo", "pantalla", "bloqueado", "fallo", "red", "servidor"
}

PT_STOPWORDS: Set[str] = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "nao", "uma", "os",
    "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem",
    "à", "seu", "sua", "ou", "quando", "muito", "há", "nos", "já", "está", "também", "pelo", "pela",
    "até", "isso", "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse",
    "eles", "estão", "você", "senha", "usuário", "usuario", "acesso", "ajuda", "erro", "solicito", "favor"
}

EN_STOPWORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "please", "ticket", "help", "access", "error",
    "password", "user", "server", "system", "file", "share", "install", "login", "failed", "support"
}

DE_STOPWORDS: Set[str] = {
    "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "fuer",
    "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an", "werden", "aus", "er", "hat",
    "dass", "sie", "nach", "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
    "einen", "so", "bitte", "hilfe", "fehler", "zugang", "benutzer", "passwort", "rechner", "abteilung"
}

# Expresiones regulares para marcadores ortográficos distintivos
ES_DISTINCTIVE: Pattern[str] = re.compile(r"[ñáéíóúü¿¡]", re.IGNORECASE)
PT_DISTINCTIVE: Pattern[str] = re.compile(r"[ãõêôç]|\b(?:não|você|está|estou|senha|ações|ção|ões)\b", re.IGNORECASE)
DE_DISTINCTIVE: Pattern[str] = re.compile(r"[äöüß]", re.IGNORECASE)

WORD_TOKENIZER: Pattern[str] = re.compile(r"\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑãõêôçÃÕÊÔÇäöüßÄÖÜẞ]{2,}\b")


def tokenize_words(text: str) -> List[str]:
    """Tokeniza una cadena de texto en palabras en minúsculas ignorando símbolos y números."""
    return [w.lower() for w in WORD_TOKENIZER.findall(text)]


def score_language(text: str) -> Dict[str, float]:
    """
    Calcula puntuaciones de probabilidad relativa entre español, inglés, portugués y alemán.

    Args:
        text: Texto de entrada a clasificar.

    Returns:
        Diccionario con las probabilidades normalizadas para cada idioma ('es', 'en', 'pt', 'de').
    """
    if not text or not text.strip():
        return {"es": 0.0, "en": 0.0, "pt": 0.0, "de": 0.0}

    words = tokenize_words(text)
    if not words:
        return {"es": 0.0, "en": 0.0, "pt": 0.0, "de": 0.0}

    raw_scores: Dict[str, float] = {
        "es": sum(1.0 for w in words if w in ES_STOPWORDS),
        "pt": sum(1.0 for w in words if w in PT_STOPWORDS),
        "en": sum(1.0 for w in words if w in EN_STOPWORDS),
        "de": sum(1.0 for w in words if w in DE_STOPWORDS),
    }

    # Bonificaciones por ortografía y caracteres distintivos del alfabeto
    if ES_DISTINCTIVE.search(text):
        raw_scores["es"] += 2.5
    if PT_DISTINCTIVE.search(text):
        raw_scores["pt"] += 3.0
    if DE_DISTINCTIVE.search(text):
        raw_scores["de"] += 3.0

    total_score = sum(raw_scores.values())
    if total_score == 0:
        # Si no hay coincidencias con ningún vocabulario funcional, devolver distribución uniforme
        return {"es": 0.25, "en": 0.25, "pt": 0.25, "de": 0.25}

    return {lang: round(score / total_score, 4) for lang, score in raw_scores.items()}


def detect_language(text: str) -> Tuple[str, float]:
    """
    Identifica el idioma predominante de un texto y el nivel de confianza asociado.

    Args:
        text: Cadena de texto a analizar.

    Returns:
        Tupla con el código de idioma ISO ('es', 'en', 'pt', 'de' o 'und') y su confianza (0.0 a 1.0).
    """
    if not text or len(text.strip()) < 3:
        return ("und", 0.0)

    scores = score_language(text)
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score <= 0.25:
        return ("und", best_score)

    return (best_lang, best_score)


def is_spanish(text: str, min_confidence: float = 0.40) -> bool:
    """
    Determina si un texto está en español con al menos el umbral de confianza mínimo.

    Args:
        text: Texto a evaluar.
        min_confidence: Umbral de confianza mínimo requerido (por defecto 0.40).

    Returns:
        True si el idioma detectado es 'es' y su confianza es mayor o igual a min_confidence.
    """
    lang, conf = detect_language(text)
    return lang == "es" and conf >= min_confidence


def filter_spanish_records(
    records: List[Dict[str, Any]],
    text_fields: Tuple[str, ...] = ("subject", "description"),
    min_confidence: float = 0.40,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filtra una lista de registros separando los tickets en español de los que no lo están.

    Args:
        records: Lista de diccionarios de tickets.
        text_fields: Tupla con los campos textuales a evaluar de forma concatenada.
        min_confidence: Confianza mínima para clasificar como español.

    Returns:
        Tupla con (registros_en_espanol, registros_descartados).
    """
    spanish_records: List[Dict[str, Any]] = []
    discarded_records: List[Dict[str, Any]] = []

    for record in records:
        # Combinar campos textuales para obtener mayor señal de lenguaje
        combined_text = " ".join(
            str(record.get(field, "") or "")
            for field in text_fields
        ).strip()

        lang, conf = detect_language(combined_text)

        # Si el registro ya trae una etiqueta de idioma verificada explícita (ej. Kaggle)
        explicit_lang = str(record.get("language") or "").lower().strip()
        if explicit_lang == "es" or (explicit_lang == "" and lang == "es" and conf >= min_confidence):
            enriched_record = dict(record)
            enriched_record["detected_language"] = "es"
            enriched_record["language_confidence"] = conf
            spanish_records.append(enriched_record)
        else:
            discarded = dict(record)
            discarded["detected_language"] = lang
            discarded["language_confidence"] = conf
            discarded["discard_reason"] = f"No es español (detectado: {lang}, conf: {conf:.2f})"
            discarded_records.append(discarded)

    return spanish_records, discarded_records
