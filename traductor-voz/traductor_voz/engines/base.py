"""Interfaz común de los motores de voz."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VoiceRef:
    """Un trozo de tu voz que el motor imita: timbre, entonación y emoción."""

    wav: np.ndarray
    sr: int
    text: str = ""  # lo que se dice en el audio (Qwen3-TTS lo necesita para clonar mejor)

    @property
    def duration(self) -> float:
        return len(self.wav) / self.sr


@dataclass
class VoiceStyle:
    exaggeration: float = 0.5  # intensidad emocional (Chatterbox)
    cfg_weight: float = 0.3  # 0 = acento nativo del idioma destino; 0.5 = conserva más tu acento
    temperature: float = 0.8


class TTSEngine(ABC):
    name: str
    languages: frozenset[str]

    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        ref: VoiceRef,
        style: VoiceStyle,
        profile: VoiceRef | None = None,
    ) -> tuple[np.ndarray, int]:
        """Dice `text` en `language` con tu voz.

        `ref` es el audio original de esa misma frase (de ahí sale la emoción del momento).
        `profile` es una grabación limpia de tu voz guardada aparte (de ahí sale el timbre, si hay).
        Devuelve (audio, frecuencia de muestreo).
        """


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
