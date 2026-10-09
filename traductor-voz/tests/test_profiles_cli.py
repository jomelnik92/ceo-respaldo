import pytest

from traductor_voz import audio, cli
from traductor_voz.engines.translate import MARIAN_PAIRS, NLLB_MODEL, choose_model
from traductor_voz.profiles import list_profiles, load_profile, save_profile

from .conftest import FakeASR, FakeTranslator, FakeTTS, voice

SR = 24000


def test_profiles(tmp_path, data_dir):
    rec = audio.save_audio(tmp_path / "lectura.wav", voice(SR, 25.0), SR)
    path = save_profile("Mi voz!", rec, FakeASR([(0, 5, " Hola"), (5, 9, " qué tal")]))
    assert path.stem == "Mi voz"
    assert list_profiles() == ["Mi voz"]
    ref = load_profile("Mi voz")
    assert ref.text == "Hola qué tal"
    assert ref.duration == pytest.approx(20.0, abs=0.05)  # se recorta a 20 s

    short = audio.save_audio(tmp_path / "corta.wav", voice(SR, 2.0), SR)
    with pytest.raises(ValueError, match="al menos"):
        save_profile("corta", short)
    with pytest.raises(ValueError, match="nombre"):
        save_profile("!!!", rec)
    with pytest.raises(FileNotFoundError):
        load_profile("otra")


def test_choose_translation_model():
    assert choose_model("es", "fr") == ("marian", "Helsinki-NLP/opus-mt-es-fr")
    assert choose_model("es", "de", "nllb") == ("nllb", NLLB_MODEL)
    assert choose_model("it", "de") == ("nllb", NLLB_MODEL)
    assert ("es", "en") in MARIAN_PAIRS and ("es", "es") not in MARIAN_PAIRS
    with pytest.raises(ValueError):
        choose_model("it", "de", "marian")
    with pytest.raises(ValueError):
        choose_model("xx", "fr")


def test_cli_translate(tmp_path, data_dir, monkeypatch, capsys):
    from traductor_voz.pipeline import VoiceTranslator

    rec = audio.save_audio(tmp_path / "nota.wav", voice(SR, 3.0), SR)
    fake = VoiceTranslator(tts=FakeTTS(), asr=FakeASR([(0.0, 3.0, " Hola a todos.")]), translator=FakeTranslator())
    monkeypatch.setattr(cli, "_translator", lambda args: fake)
    code = cli.main(["traducir", str(rec), "-i", "fr", "en", "-o", str(tmp_path / "out"), "-f", "mp3"])
    assert code == 0
    printed = capsys.readouterr().out.split()
    assert [p.rsplit("/", 1)[-1] for p in printed] == ["nota.fr.mp3", "nota.en.mp3"]
    assert audio.load_audio(printed[0], 16000).size > 0


def test_cli_errors(tmp_path, data_dir, capsys):
    assert cli.main(["perfil", "listar"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["perfil", "crear"])
    assert cli.main(["traducir", str(tmp_path / "x.wav"), "--perfil", "nadie"]) == 1
    assert "No existe el perfil" in capsys.readouterr().err
