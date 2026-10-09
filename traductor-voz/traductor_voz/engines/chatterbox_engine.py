"""Chatterbox Multilingual (Resemble AI, licencia MIT): 23 idiomas, clonación con 5-10 s de voz
y control de intensidad emocional ('exaggeration').

Truco que usamos: Chatterbox recibe la voz de referencia en dos partes.
  - Timbre (embedding del hablante + referencia del decodificador S3Gen): sale de tu perfil de voz
    si lo tienes (grabación limpia), o si no, del propio audio.
  - Prompt de tokens de habla (lo que el modelo usa para copiar ritmo, entonación y emoción):
    sale SIEMPRE del audio original de la frase que se está traduciendo.
Así cada frase traducida hereda la emoción de esa misma frase, pero con tu timbre limpio.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import TTSEngine, VoiceRef, VoiceStyle, pick_device

LANGUAGES = frozenset(
    "ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split()
)
CACHE_SIZE = 64


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    languages = LANGUAGES

    def __init__(self, device: str | None = None, ckpt_dir: str | None = None):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.device = device or pick_device()
        if ckpt_dir:
            self.model = ChatterboxMultilingualTTS.from_local(ckpt_dir, self.device)
        else:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self._conds: OrderedDict[str, object] = OrderedDict()

    def _conditionals(self, ref: VoiceRef):
        import torch

        key = hashlib.sha1(ref.wav.tobytes()).hexdigest() + f":{ref.sr}"
        if key in self._conds:
            self._conds.move_to_end(key)
            return self._conds[key]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.wav"
            sf.write(path, ref.wav, ref.sr)
            with torch.inference_mode():
                self.model.prepare_conditionals(str(path))
        self._conds[key] = self.model.conds
        if len(self._conds) > CACHE_SIZE:
            self._conds.popitem(last=False)
        return self.model.conds

    def synthesize(
        self,
        text: str,
        language: str,
        ref: VoiceRef,
        style: VoiceStyle,
        profile: VoiceRef | None = None,
    ) -> tuple[np.ndarray, int]:
        import torch
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.mtl_tts import Conditionals

        moment = self._conditionals(ref)
        timbre = self._conditionals(profile) if profile is not None else moment
        t3 = T3Cond(
            speaker_emb=timbre.t3.speaker_emb,
            cond_prompt_speech_tokens=moment.t3.cond_prompt_speech_tokens,
            emotion_adv=style.exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.model.conds = Conditionals(t3, timbre.gen)
        wav = self.model.generate(
            text,
            language_id=language,
            exaggeration=style.exaggeration,
            cfg_weight=style.cfg_weight,
            temperature=style.temperature,
        )
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), self.model.sr
