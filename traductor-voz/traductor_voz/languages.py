"""Idiomas soportados."""

# Idiomas de destino que ofrece la app (código ISO 639-1 -> nombre).
TARGETS = {
    "fr": "Francés",
    "de": "Alemán",
    "en": "Inglés",
    "es": "Español",
}

# Idiomas que se pueden reconocer como origen (Whisper entiende muchos más).
SOURCES = {
    "es": "Español",
    "en": "Inglés",
    "fr": "Francés",
    "de": "Alemán",
    "it": "Italiano",
    "pt": "Portugués",
}


def name(code: str) -> str:
    return TARGETS.get(code) or SOURCES.get(code) or code
