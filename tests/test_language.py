"""
Pruebas unitarias para el módulo de detección y filtrado de idioma español (src/preprocessing/language.py).
"""

from src.preprocessing.language import (
    detect_language,
    filter_spanish_records,
    is_spanish,
    score_language,
)


def test_detect_spanish_tickets():
    """Verifica la detección correcta de tickets escritos en español."""
    samples = [
        "No puedo conectarme a la VPN corporativa desde esta mañana.",
        "Solicito restablecer mi contraseña de usuario para el sistema ERP.",
        "La pantalla del equipo se queda en negro y arroja un error de memoria.",
        "Por favor brindar permisos de acceso a la carpeta compartida de finanzas.",
    ]
    for text in samples:
        lang, conf = detect_language(text)
        assert lang == "es", f"Fallo al detectar español en: '{text}' (detectado: {lang})"
        assert is_spanish(text) is True


def test_detect_english_tickets():
    """Verifica que tickets en inglés sean clasificados como 'en' y rechazados por is_spanish."""
    samples = [
        "File Share Access - A Support Ticket was forwarded to your role.",
        "Java installation failed on the production server.",
        "User is unable to log in with active directory credentials.",
    ]
    for text in samples:
        lang, conf = detect_language(text)
        assert lang == "en", f"Fallo al detectar inglés en: '{text}' (detectado: {lang})"
        assert is_spanish(text) is False


def test_detect_portuguese_tickets():
    """Verifica la distinción rigurosa entre portugués y español."""
    samples = [
        "Não consigo acessar o sistema com minha senha de usuário.",
        "Solicito permissão de acesso para o diretório compartilhado de arquivos.",
        "O servidor está apresentando lentidão após a atualização do banco.",
    ]
    for text in samples:
        lang, conf = detect_language(text)
        assert lang == "pt", f"Fallo al detectar portugués en: '{text}' (detectado: {lang})"
        assert is_spanish(text) is False


def test_detect_german_tickets():
    """Verifica que tickets en alemán sean clasificados como 'de'."""
    samples = [
        "Bitte um Freischaltung des Benutzerkontos für das System.",
        "Fehler beim Zugriff auf die Datenbank des Servers.",
    ]
    for text in samples:
        lang, conf = detect_language(text)
        assert lang == "de", f"Fallo al detectar alemán en: '{text}' (detectado: {lang})"
        assert is_spanish(text) is False


def test_score_language_distribution():
    """Verifica que el score sume 1.0 y contenga todas las claves requeridas."""
    text = "Problema con la cuenta de usuario"
    scores = score_language(text)
    assert set(scores.keys()) == {"es", "en", "pt", "de"}
    assert round(sum(scores.values()), 2) == 1.0
    assert scores["es"] > scores["en"]


def test_detect_empty_or_short_text():
    """Verifica que textos vacíos o excesivamente cortos retornen 'und'."""
    assert detect_language("") == ("und", 0.0)
    assert detect_language("   ") == ("und", 0.0)
    assert detect_language("ab") == ("und", 0.0)
    assert is_spanish("") is False


def test_filter_spanish_records():
    """Verifica la separación de una colección mixta de registros en español y descartados."""
    records = [
        {
            "ticket_id": "ES-01",
            "subject": "Fallo de red",
            "description": "No tengo acceso a internet ni al servidor local.",
        },
        {
            "ticket_id": "EN-01",
            "subject": "Password reset",
            "description": "Please reset my windows domain credentials.",
        },
        {
            "ticket_id": "PT-01",
            "subject": "Acesso negado",
            "description": "Não consigo conectar no servidor de arquivos.",
        },
        {
            "ticket_id": "ES-02",
            "subject": "Error en base de datos",
            "description": "La consulta reporta timeout en el módulo de clientes.",
            "language": "es",
        },
    ]

    spanish, discarded = filter_spanish_records(records, text_fields=("subject", "description"))

    assert len(spanish) == 2
    assert len(discarded) == 2

    spanish_ids = [r["ticket_id"] for r in spanish]
    assert "ES-01" in spanish_ids
    assert "ES-02" in spanish_ids
    assert all(r["detected_language"] == "es" for r in spanish)

    discarded_ids = [r["ticket_id"] for r in discarded]
    assert "EN-01" in discarded_ids
    assert "PT-01" in discarded_ids
    assert all("discard_reason" in r for r in discarded)
