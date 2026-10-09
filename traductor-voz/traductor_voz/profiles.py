"""Perfiles de voz: una grabación limpia de tu voz (10-20 s) que se guarda una vez y se reutiliza.

Con un perfil, el timbre sale de esa grabación limpia y la emoción sigue saliendo de cada frase
del audio que traduces. Sin perfil, todo sale del audio que traduces.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .audio import load_audio, resample, save_audio, trim_silence
from .engines.asr import ASR_SR
from .engines.base import VoiceRef

PROFILE_SR = 24000
MIN_SEC = 5.0
MAX_SEC = 20.0


def data_dir() -> Path:
    return Path(os.environ.get("TRADUCTOR_VOZ_DATOS", Path.home() / ".traductor-voz"))


def profiles_dir() -> Path:
    return data_dir() / "perfiles"


def _clean_name(name: str) -> str:
    clean = re.sub(r"[^\w\- ]", "", name).strip()
    if not clean:
        raise ValueError("Ponle un nombre al perfil (letras, números, espacios o guiones).")
    return clean


def save_profile(name: str, audio_path: str | Path, asr=None) -> Path:
    """Guarda el perfil. Si se pasa un reconocedor, guarda también la transcripción."""
    name = _clean_name(name)
    wav = trim_silence(load_audio(audio_path, PROFILE_SR), PROFILE_SR)
    if len(wav) < MIN_SEC * PROFILE_SR:
        raise ValueError(f"Graba al menos {MIN_SEC:.0f} segundos de voz (mejor 10-20 s, sin ruido de fondo).")
    wav = wav[: int(MAX_SEC * PROFILE_SR)]
    folder = profiles_dir()
    path = save_audio(folder / f"{name}.wav", wav, PROFILE_SR)
    text = ""
    if asr is not None:
        segments, _ = asr.transcribe(resample(wav, PROFILE_SR, ASR_SR))
        text = " ".join(t.strip() for _, _, t in segments)
    (folder / f"{name}.txt").write_text(text, encoding="utf-8")
    return path


def load_profile(name: str) -> VoiceRef:
    name = _clean_name(name)
    path = profiles_dir() / f"{name}.wav"
    if not path.exists():
        raise FileNotFoundError(f"No existe el perfil '{name}'.")
    txt = path.with_suffix(".txt")
    text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    return VoiceRef(load_audio(path, PROFILE_SR), PROFILE_SR, text)


def list_profiles() -> list[str]:
    folder = profiles_dir()
    return sorted(p.stem for p in folder.glob("*.wav")) if folder.exists() else []
