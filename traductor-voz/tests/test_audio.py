import numpy as np
import pytest

from traductor_voz import audio
from traductor_voz.prosody import estimate_f0

from .conftest import voice

SR = 24000


def median_f0(wav):
    return float(np.median(estimate_f0(wav, SR)))


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".ogg"])
def test_save_and_load_roundtrip(tmp_path, ext):
    wav = voice(SR, 1.0, 200.0)
    path = audio.save_audio(tmp_path / f"x{ext}", wav, SR)
    back = audio.load_audio(path, 16000)
    assert abs(len(back) / 16000 - 1.0) < 0.1
    assert abs(float(np.median(estimate_f0(back, 16000))) - 200.0) < 5


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg"):
        audio.load_audio(tmp_path / "no-existe.wav", SR)


@pytest.mark.parametrize("rubberband", [True, False])
def test_tempo_and_pitch(monkeypatch, rubberband):
    if rubberband and not audio.has_rubberband():
        pytest.skip("ffmpeg sin rubberband")
    monkeypatch.setattr(audio, "has_rubberband", lambda: rubberband)
    wav = voice(SR, 2.0, 150.0)

    faster = audio.change_tempo_pitch(wav, SR, tempo=1.25)
    assert abs(len(faster) / len(wav) - 1 / 1.25) < 0.03
    assert abs(median_f0(faster) - 150.0) < 4

    higher = audio.change_tempo_pitch(wav, SR, semitones=3.0)
    assert abs(len(higher) / len(wav) - 1.0) < 0.03
    assert abs(12 * np.log2(median_f0(higher) / 150.0) - 3.0) < 0.4


def test_no_change_returns_same_audio():
    wav = voice(SR, 0.5)
    assert audio.change_tempo_pitch(wav, SR) is wav


def test_trim_silence():
    wav = np.concatenate([audio.silence(SR, 0.5), voice(SR, 1.0), audio.silence(SR, 0.7)])
    trimmed = audio.trim_silence(wav, SR)
    assert 1.0 <= len(trimmed) / SR <= 1.15
    assert len(audio.trim_silence(audio.silence(SR, 1.0), SR)) <= SR


def test_match_level():
    quiet = voice(SR, 1.0, amp=0.15)
    loud_db = audio.active_level_db(voice(SR, 1.0, amp=0.4), SR)
    out = audio.match_level(quiet, SR, loud_db)
    assert abs(audio.active_level_db(out, SR) - loud_db) < 0.5
    half = audio.match_level(quiet, SR, loud_db, strength=0.5)
    assert audio.active_level_db(quiet, SR) < audio.active_level_db(half, SR) < loud_db
    # La ganancia tiene un tope para no amplificar ruido.
    whisper = voice(SR, 1.0, amp=0.01)
    capped = audio.match_level(whisper, SR, loud_db, max_gain_db=12.0)
    assert audio.active_level_db(capped, SR) == pytest.approx(audio.active_level_db(whisper, SR) + 12.0, abs=0.1)


def test_limit_peak_and_fade():
    wav = np.ones(SR, dtype=np.float32) * 1.5
    assert np.max(np.abs(audio.limit_peak(wav))) <= 0.98 + 1e-6
    faded = audio.fade(np.ones(SR, dtype=np.float32), SR)
    assert faded[0] == 0 and faded[-1] == 0 and faded[SR // 2] == 1
