"""Análisis de prosodia: tono (F0), variación de entonación, volumen y nivel de emoción.

La idea: por cada frase del audio original medimos cómo la dijiste (más aguda o más grave,
más plana o más expresiva, más fuerte o más suave). Con eso:
  - calculamos cuánta emoción pedirle al motor de voz (exaggeration en Chatterbox), y
  - corregimos la voz generada para que tenga tu mismo tono medio y tu mismo volumen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio import active_level_db, resample

F0_SR = 16000
F0_MIN = 65.0
F0_MAX = 600.0


@dataclass
class Prosody:
    duration: float  # segundos
    level_db: float  # volumen medio de la voz (dBFS)
    f0_median: float | None  # tono medio en Hz (None si no hay voz sonora)
    f0_spread: float  # variación de la entonación (desviación estándar en semitonos)
    voiced_ratio: float  # fracción del tiempo con voz sonora


def estimate_f0(wav: np.ndarray, sr: int) -> np.ndarray:
    """Devuelve las F0 (Hz) de los cuadros con voz sonora."""
    import librosa

    y = resample(wav, sr, F0_SR)
    if len(y) < 2048:
        return np.array([], dtype=np.float32)
    f0, voiced, _ = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=F0_SR, frame_length=1024, hop_length=160)
    return f0[voiced & np.isfinite(f0)]


def analyze(wav: np.ndarray, sr: int) -> Prosody:
    duration = len(wav) / sr
    if len(wav) == 0:
        return Prosody(0.0, -120.0, None, 0.0, 0.0)
    f0 = estimate_f0(wav, sr)
    n_frames = max(1, len(wav) * F0_SR // sr // 160)
    if len(f0) >= 5:
        st = 12.0 * np.log2(f0 / np.median(f0))
        median, spread = float(np.median(f0)), float(np.std(st))
    else:
        median, spread = None, 0.0
    return Prosody(
        duration=duration,
        level_db=active_level_db(wav, sr),
        f0_median=median,
        f0_spread=spread,
        voiced_ratio=min(1.0, len(f0) / n_frames),
    )


def baseline(items: list[Prosody]) -> Prosody:
    """Tu forma 'normal' de hablar en esta grabación (mediana de todas las frases)."""
    voiced = [p for p in items if p.f0_median]
    return Prosody(
        duration=float(sum(p.duration for p in items)),
        level_db=float(np.median([p.level_db for p in items])) if items else -120.0,
        f0_median=float(np.median([p.f0_median for p in voiced])) if voiced else None,
        f0_spread=float(np.median([p.f0_spread for p in voiced])) if voiced else 0.0,
        voiced_ratio=float(np.median([p.voiced_ratio for p in items])) if items else 0.0,
    )


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def arousal(p: Prosody, base: Prosody | None = None) -> float:
    """Nivel de activación emocional entre 0 (calmado, plano) y 1 (muy expresivo).

    Combina tres señales que delatan la emoción en cualquier idioma:
      - variación de la entonación (hablar 'cantando' vs. monótono),
      - volumen respecto a tu volumen habitual,
      - tono respecto a tu tono habitual (subimos el tono al emocionarnos).
    """
    spread = _clip01((p.f0_spread - 1.5) / 3.5)
    level = pitch = 0.5
    if base is not None:
        level = _clip01(0.5 + (p.level_db - base.level_db) / 12.0)
        if p.f0_median and base.f0_median:
            pitch = _clip01(0.5 + semitones(base.f0_median, p.f0_median) / 8.0)
    return 0.5 * spread + 0.3 * level + 0.2 * pitch


def exaggeration(arousal_value: float, intensity: float = 0.5) -> float:
    """Convierte la activación en el parámetro 'exaggeration' de Chatterbox.

    Con intensidad 0.5 y una frase neutra (activación 0.5) da 0.5, el valor por defecto del modelo.
    """
    return float(min(2.0, max(0.25, intensity * (0.6 + 0.8 * arousal_value))))


def semitones(f_from: float, f_to: float) -> float:
    """Distancia en semitonos de f_from a f_to."""
    return 12.0 * math.log2(f_to / f_from)


def describe(a: float) -> str:
    if a < 0.3:
        return "calmado"
    if a < 0.5:
        return "neutro"
    if a < 0.7:
        return "animado"
    return "muy expresivo"
