"""El proceso completo: audio -> texto -> traducción -> tu voz en otro idioma.

1. Whisper transcribe el audio y lo divide en frases con sus tiempos.
2. Por cada frase se mide la prosodia: tono, variación de entonación, volumen y duración.
3. Se traduce el texto de cada frase.
4. El motor de voz dice cada frase traducida con tu voz, usando como referencia el audio
   original de esa misma frase (así copia la emoción de ese momento) y una intensidad
   emocional calculada a partir de la prosodia.
5. A la voz generada se le corrige el tono medio, el ritmo y el volumen para que se parezcan a
   cómo dijiste esa frase, y se vuelven a poner tus pausas entre frases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import prosody as pr
from .audio import (
    change_tempo_pitch,
    fade,
    limit_peak,
    load_audio,
    match_level,
    resample,
    silence,
    trim_silence,
)
from .engines import TTSEngine, VoiceRef, VoiceStyle, create_tts
from .engines.asr import ASR_SR
from .segments import Phrase, merge_segments, reference_window, split_text

WORK_SR = 24000

Progress = Callable[[float, str], None]


@dataclass
class Options:
    intensity: float = 0.5  # intensidad emocional base (0.25 = contenida, 1.0 = muy dramática)
    accent: float = 0.3  # 0 = acento nativo del idioma destino, 0.5 = conserva más tu acento
    pitch_match: float = 1.0  # cuánto copiar tu tono medio en cada frase (0 a 1)
    timing_match: float = 0.5  # cuánto acercar la duración de cada frase a la original (0 a 1)
    level_match: float = 1.0  # cuánto copiar tu volumen en cada frase (0 a 1)
    keep_pauses: bool = True  # respetar tus pausas entre frases
    min_ref_sec: float = 6.0  # largo mínimo del audio de referencia de cada frase
    temperature: float = 0.8
    max_pitch_shift: float = 3.0  # semitonos
    max_speedup: float = 1.25
    max_slowdown: float = 0.85


@dataclass
class Transcript:
    wav: np.ndarray  # audio original, mono, WORK_SR
    sr: int
    language: str
    phrases: list[Phrase]
    prosody: list[pr.Prosody]
    baseline: pr.Prosody

    def reference(self, i: int, min_sec: float) -> VoiceRef:
        lo, hi = reference_window(self.phrases, i, min_sec)
        start = int(self.phrases[lo].start * self.sr)
        end = int(self.phrases[hi].end * self.sr)
        text = " ".join(p.text for p in self.phrases[lo : hi + 1])
        return VoiceRef(self.wav[start:end], self.sr, text)


@dataclass
class PhraseReport:
    index: int
    original: str
    translation: str
    emotion: str
    arousal: float
    exaggeration: float
    tempo: float
    semitones: float


@dataclass
class Result:
    language: str
    wav: np.ndarray
    sr: int
    texts: list[str]
    report: list[PhraseReport] = field(default_factory=list)


class VoiceTranslator:
    """Junta los tres motores. Los modelos se cargan recién cuando se necesitan."""

    def __init__(
        self,
        tts: str | TTSEngine = "chatterbox",
        asr=None,
        translator=None,
        whisper_model: str | None = None,
        translation_engine: str = "auto",
        tts_kwargs: dict | None = None,
    ):
        self._tts = tts
        self._tts_kwargs = tts_kwargs or {}
        self._asr = asr
        self._translator = translator
        self._whisper_model = whisper_model
        self._translation_engine = translation_engine

    @property
    def tts(self) -> TTSEngine:
        if isinstance(self._tts, str):
            self._tts = create_tts(self._tts, **self._tts_kwargs)
        return self._tts

    @property
    def asr(self):
        if self._asr is None:
            from .engines.asr import WhisperASR

            self._asr = WhisperASR(self._whisper_model)
        return self._asr

    @property
    def translator(self):
        if self._translator is None:
            from .engines.translate import Translator

            self._translator = Translator(self._translation_engine)
        return self._translator

    # 1-2. Transcribir y medir la prosodia
    def transcribe(self, path: str | Path, language: str | None = None) -> Transcript:
        wav16 = load_audio(path, ASR_SR)
        if len(wav16) < ASR_SR * 0.3:
            raise ValueError("El audio está vacío o es demasiado corto.")
        segments, detected = self.asr.transcribe(wav16, language)
        phrases = merge_segments(segments)
        if not phrases:
            raise ValueError("No se entendió ninguna palabra en el audio.")
        wav = load_audio(path, WORK_SR)
        prosody = [
            pr.analyze(trim_silence(wav[int(p.start * WORK_SR) : int(p.end * WORK_SR)], WORK_SR), WORK_SR)
            for p in phrases
        ]
        return Transcript(wav, WORK_SR, language or detected, phrases, prosody, pr.baseline(prosody))

    # 3. Traducir
    def translate(self, tr: Transcript, target: str) -> list[str]:
        return self.translator.translate([p.text for p in tr.phrases], tr.language, target)

    # 4-5. Hablar con tu voz
    def synthesize(
        self,
        tr: Transcript,
        target: str,
        texts: list[str],
        opts: Options | None = None,
        profile: VoiceRef | None = None,
        progress: Progress | None = None,
    ) -> Result:
        opts = opts or Options()
        if target not in self.tts.languages:
            raise ValueError(f"El motor {self.tts.name} no habla '{target}'.")
        if len(texts) != len(tr.phrases):
            raise ValueError("Tiene que haber un texto traducido por cada frase.")

        pieces: list[np.ndarray] = []
        report: list[PhraseReport] = []
        prev_end = 0.0
        for i, (phrase, text) in enumerate(zip(tr.phrases, texts)):
            if progress:
                progress(i / len(texts), f"Frase {i + 1} de {len(texts)}")
            text = text.strip()
            if not text:
                continue
            gap = phrase.start - prev_end
            if i == 0:
                gap = min(max(gap, 0.1), 0.5)
            elif opts.keep_pauses:
                gap = min(max(gap, 0.12), 2.0)
            else:
                gap = 0.35
            prev_end = phrase.end

            p = tr.prosody[i]
            a = pr.arousal(p, tr.baseline)
            style = VoiceStyle(pr.exaggeration(a, opts.intensity), opts.accent, opts.temperature)
            wav = self._speak(text, target, tr.reference(i, opts.min_ref_sec), style, profile)
            wav, tempo, st = self._match_prosody(wav, p, opts)

            pieces += [silence(WORK_SR, gap), wav]
            report.append(
                PhraseReport(i + 1, phrase.text, text, pr.describe(a), a, style.exaggeration, tempo, st)
            )

        pieces.append(silence(WORK_SR, 0.3))
        out = limit_peak(np.concatenate(pieces).astype(np.float32))
        if progress:
            progress(1.0, "Listo")
        return Result(target, out, WORK_SR, list(texts), report)

    def run(
        self,
        path: str | Path,
        targets: list[str],
        opts: Options | None = None,
        profile: VoiceRef | None = None,
        source_language: str | None = None,
    ) -> tuple[Transcript, dict[str, Result]]:
        tr = self.transcribe(path, source_language)
        results = {}
        for target in targets:
            results[target] = self.synthesize(tr, target, self.translate(tr, target), opts, profile)
        return tr, results

    def _speak(self, text: str, target: str, ref: VoiceRef, style: VoiceStyle, profile: VoiceRef | None) -> np.ndarray:
        parts: list[np.ndarray] = []
        for chunk in split_text(text):
            if parts:
                parts.append(silence(WORK_SR, 0.15))
            wav, sr = self.tts.synthesize(chunk, target, ref, style, profile)
            parts.append(resample(np.asarray(wav, dtype=np.float32), sr, WORK_SR))
        return np.concatenate(parts) if parts else silence(WORK_SR, 0)

    def _match_prosody(self, wav: np.ndarray, p: pr.Prosody, opts: Options) -> tuple[np.ndarray, float, float]:
        """Corrige tono medio, duración y volumen de la frase generada para que se parezca a la tuya."""
        wav = trim_silence(wav, WORK_SR)
        if len(wav) == 0:
            return wav, 1.0, 0.0

        st = 0.0
        if opts.pitch_match > 0 and p.f0_median:
            f0 = pr.estimate_f0(wav, WORK_SR)
            if len(f0) >= 5:
                diff = pr.semitones(float(np.median(f0)), p.f0_median)
                diff = max(-opts.max_pitch_shift, min(opts.max_pitch_shift, diff))
                st = diff * opts.pitch_match if abs(diff) >= 0.3 else 0.0

        tempo = 1.0
        if opts.timing_match > 0 and p.duration > 0.3:
            ratio = (len(wav) / WORK_SR) / p.duration  # >1: la traducción quedó más larga
            tempo = max(opts.max_slowdown, min(opts.max_speedup, ratio**opts.timing_match))

        wav = change_tempo_pitch(wav, WORK_SR, tempo, st)
        wav = match_level(wav, WORK_SR, p.level_db, opts.level_match)
        return fade(wav, WORK_SR), tempo, st
