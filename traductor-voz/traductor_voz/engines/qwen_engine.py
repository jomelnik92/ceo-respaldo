"""Qwen3-TTS Base (Alibaba, licencia Apache 2.0): 10 idiomas, clonación con 3 s de voz.

Clona por 'aprendizaje en contexto': recibe tu audio y su transcripción, y continúa hablando
con tu voz y con la misma forma de decir las cosas. Por eso le pasamos como referencia el audio
original de cada frase junto con lo que dijiste en ella.
No tiene un control explícito de emoción: la emoción sale toda de la referencia, así que
'Intensidad emocional' y 'Acento' no lo afectan (sí el ajuste de tono, ritmo y volumen).
"""

from __future__ import annotations

import numpy as np

from .base import TTSEngine, VoiceRef, VoiceStyle, pick_device

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MIN_REF_SEC = 3.0


class QwenEngine(TTSEngine):
    name = "qwen3"
    languages = frozenset(LANGUAGE_NAMES)

    def __init__(self, device: str | None = None, model_id: str = DEFAULT_MODEL):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.device = device or pick_device()
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        device_map = "cuda:0" if self.device == "cuda" else self.device
        self.model = Qwen3TTSModel.from_pretrained(model_id, device_map=device_map, dtype=dtype)

    def synthesize(
        self,
        text: str,
        language: str,
        ref: VoiceRef,
        style: VoiceStyle,
        profile: VoiceRef | None = None,
    ) -> tuple[np.ndarray, int]:
        use = ref
        if profile is not None and (ref.duration < MIN_REF_SEC or not ref.text):
            use = profile
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=LANGUAGE_NAMES[language],
            ref_audio=(use.wav, use.sr),
            ref_text=use.text or None,
            x_vector_only_mode=not use.text,
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)
