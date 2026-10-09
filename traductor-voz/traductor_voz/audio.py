"""Utilidades de audio: lectura y escritura con ffmpeg, y efectos de tempo, tono y volumen.

Todo el audio interno es mono, float32, con valores entre -1 y 1.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_SEC = 0.025
HOP_SEC = 0.010


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("No se encontró ffmpeg. Instálalo y agrégalo al PATH.")
    return exe


def _run_ffmpeg(args: list[str], stdin: bytes | None = None) -> bytes:
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, input=stdin, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Lee cualquier formato que entienda ffmpeg (wav, mp3, ogg/opus de WhatsApp, m4a...)."""
    raw = _run_ffmpeg(["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32).copy()


_CODECS = {
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    ".ogg": ["-c:a", "libopus", "-b:a", "64k"],
    ".opus": ["-c:a", "libopus", "-b:a", "64k"],
    ".m4a": ["-c:a", "aac", "-b:a", "128k"],
}


def save_audio(path: str | Path, wav: np.ndarray, sr: int) -> Path:
    """Guarda en wav directamente; para otros formatos (mp3, ogg, m4a) usa ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    if path.suffix.lower() == ".wav":
        sf.write(path, wav, sr, subtype="PCM_16")
        return path
    codec = _CODECS.get(path.suffix.lower(), [])
    _run_ffmpeg(
        ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", *codec, "-y", str(path)],
        stdin=wav.tobytes(),
    )
    return path


def apply_filter(wav: np.ndarray, sr: int, af: str) -> np.ndarray:
    """Pasa el audio por un filtro de ffmpeg (-af) y devuelve el resultado."""
    raw = _run_ffmpeg(
        ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-af", af, "-f", "f32le", "-"],
        stdin=np.asarray(wav, dtype=np.float32).tobytes(),
    )
    return np.frombuffer(raw, dtype=np.float32).copy()


@functools.cache
def has_rubberband() -> bool:
    """Rubber Band da mejor calidad al cambiar tono/tempo y conserva los formantes (el timbre)."""
    try:
        out = _run_ffmpeg(["-filters"])
    except RuntimeError:
        return False
    return any(line.split()[1:2] == ["rubberband"] for line in out.decode().splitlines())


def change_tempo_pitch(wav: np.ndarray, sr: int, tempo: float = 1.0, semitones: float = 0.0) -> np.ndarray:
    """Cambia la velocidad (tempo > 1 acelera) y el tono (en semitonos) de forma independiente."""
    if abs(tempo - 1.0) < 1e-3 and abs(semitones) < 1e-2:
        return wav
    ratio = 2.0 ** (semitones / 12.0)
    if has_rubberband():
        af = f"rubberband=tempo={tempo:.5f}:pitch={ratio:.5f}:formant=preserved:pitchq=quality"
    else:
        # asetrate sube el tono y acelera a la vez; atempo corrige la velocidad.
        af = f"asetrate={sr * ratio:.3f},aresample={sr},atempo={tempo / ratio:.5f}"
    return apply_filter(wav, sr, af)


def resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


def frame_rms_db(wav: np.ndarray, sr: int) -> np.ndarray:
    """Energía por cuadros de 25 ms (salto de 10 ms), en dBFS."""
    frame = max(1, int(sr * FRAME_SEC))
    hop = max(1, int(sr * HOP_SEC))
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    power = np.mean(wav[idx].astype(np.float64) ** 2, axis=1)
    return 10.0 * np.log10(power + 1e-12)


def active_level_db(wav: np.ndarray, sr: int, floor_db: float = 35.0) -> float:
    """Nivel medio de la voz, ignorando silencios (cuadros a más de floor_db bajo el máximo)."""
    db = frame_rms_db(wav, sr)
    active = db[db > db.max() - floor_db]
    return float(10.0 * np.log10(np.mean(10.0 ** (active / 10.0)) + 1e-12))


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0, pad_sec: float = 0.04) -> np.ndarray:
    """Quita silencio al principio y al final."""
    if len(wav) == 0:
        return wav
    db = frame_rms_db(wav, sr)
    loud = np.flatnonzero(db > db.max() - top_db)
    if len(loud) == 0:
        return wav[:0]
    hop = int(sr * HOP_SEC)
    pad = int(sr * pad_sec)
    start = max(0, loud[0] * hop - pad)
    end = min(len(wav), loud[-1] * hop + int(sr * FRAME_SEC) + pad)
    return wav[start:end]


def match_level(wav: np.ndarray, sr: int, target_db: float, strength: float = 1.0, max_gain_db: float = 12.0) -> np.ndarray:
    """Ajusta el volumen para que suene tan fuerte (o tan suave) como el original."""
    if len(wav) == 0 or strength <= 0:
        return wav
    gain_db = np.clip((target_db - active_level_db(wav, sr)) * strength, -max_gain_db, max_gain_db)
    return (wav * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def fade(wav: np.ndarray, sr: int, sec: float = 0.01) -> np.ndarray:
    n = min(int(sr * sec), len(wav) // 2)
    if n <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def limit_peak(wav: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return wav * (peak / m) if m > peak else wav


def silence(sr: int, sec: float) -> np.ndarray:
    return np.zeros(max(0, int(round(sr * sec))), dtype=np.float32)
