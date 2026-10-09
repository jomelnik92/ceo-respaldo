import numpy as np
import pytest

from traductor_voz import prosody as pr

from .conftest import voice

SR = 24000


def test_analyze_flat_voice():
    p = pr.analyze(voice(SR, 1.5, 180.0), SR)
    assert p.f0_median == pytest.approx(180.0, abs=4)
    assert p.f0_spread < 0.5
    assert p.voiced_ratio > 0.8
    assert p.duration == pytest.approx(1.5)


def test_analyze_expressive_voice_has_more_spread():
    flat = pr.analyze(voice(SR, 1.5, 180.0), SR)
    lively = pr.analyze(voice(SR, 1.5, 180.0, vibrato_st=5.0), SR)
    assert lively.f0_spread > 2.5 > flat.f0_spread


def test_analyze_empty():
    p = pr.analyze(np.zeros(0, dtype=np.float32), SR)
    assert p.f0_median is None and p.duration == 0


def test_arousal_grows_with_expressiveness():
    calm = pr.analyze(voice(SR, 1.5, 140.0, amp=0.1), SR)
    normal = pr.analyze(voice(SR, 1.5, 150.0, vibrato_st=1.5, amp=0.2), SR)
    excited = pr.analyze(voice(SR, 1.5, 200.0, vibrato_st=5.0, amp=0.5), SR)
    base = pr.baseline([calm, normal, excited])
    a = [pr.arousal(p, base) for p in (calm, normal, excited)]
    assert a[0] < a[1] < a[2]
    assert all(0 <= x <= 1 for x in a)


def test_exaggeration_mapping():
    assert pr.exaggeration(0.5, 0.5) == pytest.approx(0.5)
    assert pr.exaggeration(0.0, 0.5) < 0.5 < pr.exaggeration(1.0, 0.5)
    assert pr.exaggeration(0.0, 0.25) == 0.25  # mínimo
    assert pr.exaggeration(1.0, 2.0) == 2.0  # máximo


def test_semitones():
    assert pr.semitones(100.0, 200.0) == pytest.approx(12.0)
    assert pr.semitones(200.0, 100.0) == pytest.approx(-12.0)
