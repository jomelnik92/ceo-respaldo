"""Prueba la app web de punta a punta (servidor real de Gradio + cliente), con motores falsos."""

import numpy as np
import pytest

from traductor_voz import audio
from traductor_voz.pipeline import VoiceTranslator

from .conftest import FakeASR, FakeTranslator, FakeTTS, voice

SR = 24000


@pytest.fixture
def client(data_dir):
    gradio_client = pytest.importorskip("gradio_client")
    from traductor_voz.app import build_ui

    segments = [(0.0, 1.5, " Hola a todos."), (2.0, 3.5, " ¡Nos vemos!")]
    vt = VoiceTranslator(tts=FakeTTS(), asr=FakeASR(segments), translator=FakeTranslator())
    demo = build_ui(vt)
    demo.queue().launch(prevent_thread_lock=True, server_name="127.0.0.1", server_port=None, quiet=True)
    try:
        yield gradio_client.Client(demo.local_url, verbose=False), gradio_client.handle_file
    finally:
        demo.close()


def has_audio(update) -> bool:
    if isinstance(update, dict):
        return bool(update.get("visible") and update.get("value"))
    return bool(update)


def test_translate_and_regenerate(client, tmp_path):
    c, handle_file = client
    wav = np.concatenate([voice(SR, 1.5), audio.silence(SR, 0.5), voice(SR, 1.5, 200, 4, 0.5)])
    rec = audio.save_audio(tmp_path / "nota.wav", wav, SR)
    sliders = [0.5, 0.3, 1.0, 0.5, True]
    out = c.predict(
        handle_file(str(rec)),
        ["fr", "en"],
        "(sin perfil: usar el mismo audio)",
        *sliders,
        "auto",
        api_name="/on_translate",
    )
    fr, de, en, es, table, analysis, status = out
    assert has_audio(fr) and has_audio(en) and not has_audio(de) and not has_audio(es)
    assert "2 frases" in status
    rows = table["data"] if isinstance(table, dict) else table
    assert rows[0][1] == "Hola a todos." and rows[0][2] == "[fr] Hola a todos."
    assert len((analysis["data"] if isinstance(analysis, dict) else analysis)) == 4

    rows[0][2] = "Bonjour à tous !"
    out = c.predict(
        {"headers": ["#", "Original", "Francés", "Alemán", "Inglés", "Español"], "data": rows, "metadata": None},
        ["fr", "de"],
        "(sin perfil: usar el mismo audio)",
        *sliders,
        api_name="/on_regenerate",
    )
    fr, de, en, es, table, _, _ = out
    assert has_audio(fr) and has_audio(de) and not has_audio(en)
    rows = table["data"] if isinstance(table, dict) else table
    assert rows[0][2] == "Bonjour à tous !"
    assert rows[0][3] == "[de] Hola a todos."
