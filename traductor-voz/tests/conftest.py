"""Motores falsos y voces sintéticas para probar el proceso sin descargar modelos."""

from __future__ import annotations

import numpy as np
import pytest

from traductor_voz.engines.base import TTSEngine


def voice(sr: int, dur: float, f0: float = 150.0, vibrato_st: float = 0.0, amp: float = 0.3) -> np.ndarray:
    """Una 'voz' sintética: tono armónico con vibrato opcional (variación de entonación)."""
    t = np.arange(int(sr * dur)) / sr
    freq = f0 * 2 ** (vibrato_st * np.sin(2 * np.pi * 3.0 * t) / 12)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    sig = sum(0.6**k * np.sin((k + 1) * phase) for k in range(5))
    env = np.minimum(1.0, np.minimum(t, dur - t) / 0.02)
    return (amp * sig / np.max(np.abs(sig)) * env).astype(np.float32)


class FakeASR:
    def __init__(self, segments, language="es"):
        self.segments = segments
        self.language = language

    def transcribe(self, wav16k, language=None):
        return list(self.segments), language or self.language


class FakeTranslator:
    def translate(self, texts, src, tgt):
        return [f"[{tgt}] {t}" for t in texts]


class FakeTTS(TTSEngine):
    name = "fake"
    languages = frozenset({"fr", "de", "en", "es"})

    def __init__(self, sr: int = 22050, f0: float = 150.0):
        self.sr = sr
        self.f0 = f0
        self.calls: list[dict] = []

    def synthesize(self, text, language, ref, style, profile=None):
        self.calls.append({"text": text, "language": language, "ref": ref, "style": style, "profile": profile})
        return voice(self.sr, 0.5 + 0.05 * len(text), self.f0, amp=0.1), self.sr


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUCTOR_VOZ_DATOS", str(tmp_path / "datos"))
    return tmp_path / "datos"
