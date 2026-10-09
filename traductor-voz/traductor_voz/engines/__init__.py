"""Motores intercambiables: reconocimiento de voz, traducción y síntesis con tu voz."""

from __future__ import annotations

from .base import TTSEngine, VoiceRef, VoiceStyle

TTS_ENGINES = ("chatterbox", "qwen3")


def create_tts(name: str, **kwargs) -> TTSEngine:
    """Crea el motor de voz. Se importa recién acá porque cada uno trae dependencias pesadas."""
    if name == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine

        return ChatterboxEngine(**kwargs)
    if name == "qwen3":
        from .qwen_engine import QwenEngine

        return QwenEngine(**kwargs)
    raise ValueError(f"Motor de voz desconocido: {name!r}. Opciones: {', '.join(TTS_ENGINES)}")


__all__ = ["TTS_ENGINES", "TTSEngine", "VoiceRef", "VoiceStyle", "create_tts"]
