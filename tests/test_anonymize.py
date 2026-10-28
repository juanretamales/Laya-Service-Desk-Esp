"""
Pruebas unitarias para el módulo de anonimización y sanitización PII (src/preprocessing/anonymize.py).
"""

from src.preprocessing.anonymize import (
    anonymize_record,
    anonymize_text,
    detect_pii,
)


def test_anonymize_email():
    """Verifica el enmascaramiento de direcciones de correo electrónico."""
    raw = "Por favor enviar respuesta a juan.perez@empresa.com o soporte.it@corp.org."
    expected = "Por favor enviar respuesta a <EMAIL> o <EMAIL>."
    assert anonymize_text(raw) == expected


def test_anonymize_ip_addresses():
    """Verifica el enmascaramiento de direcciones IPv4 e IPv6."""
    raw = "El servidor 192.168.1.150 y el router 10.0.0.1 no responden a ping."
    expected = "El servidor <IP> y el router <IP> no responden a ping."
    assert anonymize_text(raw) == expected


def test_anonymize_mac_address():
    """Verifica el enmascaramiento de direcciones físicas MAC."""
    raw = "La tarjeta con MAC 00:1A:2B:3C:4D:5E fue bloqueada en el switch."
    expected = "La tarjeta con MAC <MAC> fue bloqueada en el switch."
    assert anonymize_text(raw) == expected


def test_anonymize_phone_numbers():
    """Verifica el enmascaramiento de teléfonos en formatos locales e internacionales."""
    raw = "Contactar al +34 612 345 678 o al fijo (91) 555 4321 urgente."
    sanitized = anonymize_text(raw)
    assert "<PHONE>" in sanitized
    assert "612 345 678" not in sanitized


def test_anonymize_credit_cards():
    """Verifica el enmascaramiento de secuencias de tarjetas de 16 dígitos."""
    raw = "Error en cobro de tarjeta 4532 1234 5678 9010 en la factura #88."
    expected = "Error en cobro de tarjeta <CARD> en la factura #88."
    assert anonymize_text(raw) == expected


def test_anonymize_urls():
    """Verifica el enmascaramiento de URLs y portales web."""
    raw = "Acceder a https://intranet.miempresa.com/login para validar credenciales."
    expected = "Acceder a <URL> para validar credenciales."
    assert anonymize_text(raw) == expected


def test_anonymize_id_documents():
    """Verifica el enmascaramiento de identificadores tipo DNI/NIE."""
    raw = "El empleado con DNI 12345678Z solicita acceso a la VPN."
    expected = "El empleado con DNI <ID_DOC> solicita acceso a la VPN."
    assert anonymize_text(raw) == expected


def test_standardize_legacy_tags():
    """Verifica la estandarización de placeholders heredados de Zenodo/Kaggle."""
    raw = "File Share Access - [TICKET ID] - [NAME] ([COMPANY]. [LOCATION]) on [SERVER]"
    expected = "File Share Access - <TICKET_ID> - <NAME> (<ORG>. <LOC>) on <SERVER>"
    assert anonymize_text(raw) == expected


def test_detect_pii():
    """Verifica el recuento estadístico exacto de entidades sensibles detectadas."""
    text = "Contactar a admin@corp.com o soporte@corp.com desde 192.168.0.1"
    counts = detect_pii(text)
    assert counts == {"email": 2, "ipv4": 1}


def test_anonymize_record():
    """Verifica que anonymize_record procese solo los campos textuales especificados."""
    record = {
        "ticket_id": "TCK-100",
        "subject": "Fallo de VPN para maria@empresa.com",
        "description": "No conecta a la IP 172.16.0.4. Contacto: +34 600112233.",
        "category": "network",
        "priority": "high",
        "support_group": "network",
    }
    sanitized = anonymize_record(record, text_fields=("subject", "description"))

    assert sanitized["ticket_id"] == "TCK-100"
    assert sanitized["category"] == "network"
    assert sanitized["subject"] == "Fallo de VPN para <EMAIL>"
    assert "<IP>" in sanitized["description"]
    assert "<PHONE>" in sanitized["description"]
    assert "maria@empresa.com" not in sanitized["subject"]


def test_anonymize_empty_and_none():
    """Verifica el manejo seguro de entradas vacías o nulas."""
    assert anonymize_text("") == ""
    assert anonymize_text(None) == ""
    assert detect_pii("") == {}
