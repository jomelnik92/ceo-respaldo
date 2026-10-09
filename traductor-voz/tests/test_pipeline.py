import numpy as np
import pytest

from traductor_voz import audio
from traductor_voz import prosody as pr
from traductor_voz.engines.base import VoiceRef
from traductor_voz.pipeline import WORK_SR, Options, VoiceTranslator

from .conftest import FakeASR, FakeTranslator, FakeTTS, voice

SR = WORK_SR
SEGMENTS = [
    (0.3, 2.3, " Hola, ¿cómo estás?"),
    (2.9, 4.4, " ¡Qué alegría verte!"),
    (5.2, 7.7, " Bueno, nos vemos."),
]


@pytest.fixture
def recording(tmp_path):
    """Tres frases: neutra, emocionada (más aguda, fuerte y con entonación) y calmada."""
    wav = np.concatenate(
        [
            audio.silence(SR, 0.3),
            voice(SR, 2.0, 150.0, vibrato_st=1.0, amp=0.2),
            audio.silence(SR, 0.6),
            voice(SR, 1.5, 220.0, vibrato_st=5.0, amp=0.5),
            audio.silence(SR, 0.8),
            voice(SR, 2.5, 140.0, amp=0.12),
            audio.silence(SR, 0.4),
        ]
    )
    return audio.save_audio(tmp_path / "nota.wav", wav, SR)


@pytest.fixture
def vt():
    return VoiceTranslator(tts=FakeTTS(), asr=FakeASR(SEGMENTS), translator=FakeTranslator())


def test_transcribe_measures_each_phrase(vt, recording):
    tr = vt.transcribe(recording)
    assert tr.language == "es"
    assert [p.text for p in tr.phrases] == ["Hola, ¿cómo estás?", "¡Qué alegría verte!", "Bueno, nos vemos."]
    for p, f0 in zip(tr.prosody, [150, 220, 140]):
        assert abs(pr.semitones(f0, p.f0_median)) < 2  # el vibrato fuerte desvía un poco la mediana
    assert tr.prosody[1].level_db > tr.prosody[0].level_db > tr.prosody[2].level_db
    assert tr.prosody[0].duration == pytest.approx(2.0, abs=0.1)


def test_emotional_phrase_gets_more_intensity(vt, recording):
    tr = vt.transcribe(recording)
    res = vt.synthesize(tr, "fr", vt.translate(tr, "fr"))
    exag = [c["style"].exaggeration for c in vt.tts.calls]
    assert exag[1] > exag[0] > exag[2]
    assert [r.emotion for r in res.report][1] in ("animado", "muy expresivo")
    assert all(c["language"] == "fr" for c in vt.tts.calls)
    assert vt.tts.calls[0]["text"] == "[fr] Hola, ¿cómo estás?"


def test_reference_is_the_phrase_itself_or_its_neighbours(vt, recording):
    tr = vt.transcribe(recording)
    vt.synthesize(tr, "en", vt.translate(tr, "en"), Options(min_ref_sec=1.0))
    refs = [c["ref"] for c in vt.tts.calls]
    assert [r.text for r in refs] == [p.text for p in tr.phrases]
    assert refs[1].duration == pytest.approx(1.5, abs=0.01)

    vt.tts.calls.clear()
    vt.synthesize(tr, "en", vt.translate(tr, "en"), Options(min_ref_sec=4.0))
    ref = vt.tts.calls[1]["ref"]
    assert ref.duration >= 4.0
    assert "¡Qué alegría verte!" in ref.text


def test_pitch_timing_and_pauses(vt, recording):
    tr = vt.transcribe(recording)
    res = vt.synthesize(tr, "de", vt.translate(tr, "de"))
    st = [r.semitones for r in res.report]
    assert st[0] == 0.0  # mismo tono: no se toca
    assert st[1] == pytest.approx(3.0)  # 220 Hz vs 150 Hz: se sube hasta el límite de 3 semitonos
    assert st[2] == pytest.approx(pr.semitones(150, 140), abs=0.3)
    assert all(0.85 <= r.tempo <= 1.25 for r in res.report)

    # Las pausas originales (0.6 s y 0.8 s) aparecen en la salida.
    db = audio.frame_rms_db(res.wav, SR)
    silent = db < db.max() - 50
    runs, count = [], 0
    for s in silent:
        if s:
            count += 1
        elif count:
            runs.append(count * audio.HOP_SEC)
            count = 0
    inner = sorted(runs[1:])
    assert len(inner) == 2
    assert inner[0] == pytest.approx(0.6, abs=0.12) and inner[1] == pytest.approx(0.8, abs=0.12)


def test_match_prosody_copies_level_and_pitch(vt):
    original = pr.analyze(voice(SR, 2.0, 200.0, amp=0.5), SR)
    generated = voice(SR, 2.0, 180.0, amp=0.15)
    out, tempo, st = vt._match_prosody(generated, original, Options())
    assert st == pytest.approx(pr.semitones(180, 200), abs=0.2)
    assert abs(audio.active_level_db(out, SR) - original.level_db) < 1.0
    assert float(np.median(pr.estimate_f0(out, SR))) == pytest.approx(200, abs=5)
    assert tempo == pytest.approx(1.0, abs=0.02)


def test_no_pauses_option(vt, recording):
    tr = vt.transcribe(recording)
    with_pauses = vt.synthesize(tr, "fr", vt.translate(tr, "fr"))
    without = vt.synthesize(tr, "fr", vt.translate(tr, "fr"), Options(keep_pauses=False, timing_match=0))
    assert len(without.wav) != len(with_pauses.wav)


def test_edited_and_empty_texts(vt, recording):
    tr = vt.transcribe(recording)
    res = vt.synthesize(tr, "fr", ["Salut !", "", "À plus."])
    assert [c["text"] for c in vt.tts.calls] == ["Salut !", "À plus."]
    assert len(res.report) == 2
    with pytest.raises(ValueError):
        vt.synthesize(tr, "fr", ["solo una"])


def test_profile_is_passed_to_engine(vt, recording):
    tr = vt.transcribe(recording)
    profile = VoiceRef(voice(SR, 8.0), SR, "texto del perfil")
    vt.synthesize(tr, "fr", vt.translate(tr, "fr"), profile=profile)
    assert all(c["profile"] is profile for c in vt.tts.calls)


def test_long_translation_is_split(vt, recording):
    tr = vt.transcribe(recording)
    long_text = "Une phrase assez longue pour le test. " * 10
    vt.synthesize(tr, "fr", [long_text, "Oui.", "Non."])
    assert len(vt.tts.calls) > 3


def test_resamples_engine_output(vt, recording):
    tr = vt.transcribe(recording)
    res = vt.synthesize(tr, "fr", vt.translate(tr, "fr"))
    assert res.sr == SR
    assert res.wav.dtype == np.float32
    assert np.max(np.abs(res.wav)) <= 0.98 + 1e-6


def test_errors(vt, recording, tmp_path):
    tr = vt.transcribe(recording)
    with pytest.raises(ValueError, match="no habla"):
        vt.synthesize(tr, "ja", ["a", "b", "c"])
    empty = VoiceTranslator(tts=FakeTTS(), asr=FakeASR([]), translator=FakeTranslator())
    with pytest.raises(ValueError, match="ninguna palabra"):
        empty.transcribe(recording)
    short = audio.save_audio(tmp_path / "corto.wav", audio.silence(SR, 0.1), SR)
    with pytest.raises(ValueError, match="corto"):
        vt.transcribe(short)


def test_run_translates_to_all_targets(vt, recording):
    tr, results = vt.run(recording, ["fr", "de", "en"])
    assert set(results) == {"fr", "de", "en"}
    assert results["de"].texts[0] == "[de] Hola, ¿cómo estás?"
