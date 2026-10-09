"""Reconocimiento de voz con faster-whisper: texto con marcas de tiempo e idioma detectado."""

from __future__ import annotations

import numpy as np

ASR_SR = 16000


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class WhisperASR:
    def __init__(self, model_size: str | None = None, device: str | None = None, compute_type: str | None = None):
        from faster_whisper import WhisperModel

        cuda = _cuda_available() if device is None else device == "cuda"
        self.device = "cuda" if cuda else "cpu"
        # En GPU, large-v3-turbo es rápido y muy preciso; en CPU, 'small' es un buen equilibrio.
        self.model_size = model_size or ("large-v3-turbo" if cuda else "small")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type or ("float16" if cuda else "int8"),
        )

    def transcribe(self, wav16k: np.ndarray, language: str | None = None) -> tuple[list[tuple[float, float, str]], str]:
        """Devuelve ([(inicio, fin, texto), ...], idioma detectado)."""
        segments, info = self.model.transcribe(
            wav16k,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return [(s.start, s.end, s.text) for s in segments], info.language
