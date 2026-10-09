"""Prueba el bot de punta a punta: librería real de Telegram contra un servidor de Telegram falso."""

import asyncio
import json

import numpy as np
import pytest

from traductor_voz import audio
from traductor_voz.pipeline import VoiceTranslator

from .conftest import FakeASR, FakeTranslator, FakeTTS, voice

pytest.importorskip("telegram")
pytest.importorskip("uvicorn")

from traductor_voz.bot import SettingsStore, TranslatorBot  # noqa: E402

from .fake_telegram import FakeTelegram  # noqa: E402

SR = 24000
OWNER, STRANGER = 42, 99


def ogg(tmp_path, name, seconds):
    wav = np.concatenate([voice(SR, seconds / 2), audio.silence(SR, 0.4), voice(SR, seconds / 2, 200, 4, 0.4)])
    return audio.save_audio(tmp_path / f"{name}.ogg", wav, SR).read_bytes()


def test_bot_end_to_end(tmp_path, data_dir):
    tts = FakeTTS()
    vt = VoiceTranslator(
        tts=tts,
        asr=FakeASR([(0.0, 2.0, " Hola a todos."), (2.4, 4.4, " ¡Nos vemos!")]),
        translator=FakeTranslator(),
    )
    note = ogg(tmp_path, "nota", 4.0)
    reading = ogg(tmp_path, "lectura", 9.0)
    bot = TranslatorBot(vt, {OWNER}, SettingsStore(data_dir / "bot.json"))

    with FakeTelegram() as tg:
        app = bot.build(tg.token, base_url=f"{tg.url}/bot", base_file_url=f"{tg.url}/file/bot")

        async def said(fragment):
            await tg.wait_for(lambda: any(fragment in t for t in tg.texts()))

        async def scenario():
            async with app:
                await app.start()
                await app.updater.start_polling(poll_interval=0.05, timeout=1)
                try:
                    await bot._post_init(app)
                    assert "idiomas" in tg.sent("setMyCommands")[0]["commands"]

                    # Un desconocido: no se procesa nada y se le dice su número de usuario.
                    tg.send_voice(STRANGER, note)
                    await said(f"es {STRANGER}")

                    tg.send_text(OWNER, "/idiomas en fr")
                    await said("Idiomas: Inglés, Francés")
                    tg.send_text(OWNER, "/idiomas xx")
                    await said("No conozco: xx")
                    tg.send_text(OWNER, "/acento 0.1")
                    await said("Acento: 0.1")
                    tg.send_text(OWNER, "/intensidad 7")
                    await said("entre 0.25 y 1")

                    # Nota de voz -> una nota de voz por idioma, con el texto traducido.
                    tg.send_voice(OWNER, note)
                    await tg.wait_for(lambda: len(tg.sent("sendVoice")) == 2)
                    english, french = tg.sent("sendVoice")
                    assert english["caption"] == "Inglés: [en] Hola a todos. [en] ¡Nos vemos!"
                    assert french["caption"].startswith("Francés: [fr] Hola a todos.")
                    assert english["voice"][:4] == b"OggS"
                    assert int(english["duration"]) >= 1
                    status = tg.sent("editMessageText")[-1]["text"]
                    assert "Entendí (Español): «Hola a todos. ¡Nos vemos!»" in status
                    assert {c["style"].cfg_weight for c in tts.calls} == {0.1}
                    assert all(c["profile"] is None for c in tts.calls)

                    # Grabar un perfil de voz por el bot y usarlo.
                    tg.send_text(OWNER, "/grabarperfil yo")
                    await said("Mándame ahora una nota de voz")
                    tg.send_voice(OWNER, reading, duration=9)
                    await said("Perfil «yo» guardado")
                    tts.calls.clear()
                    tg.send_voice(OWNER, note)
                    await tg.wait_for(lambda: len(tg.sent("sendVoice")) == 4)
                    assert tts.calls and all(c["profile"] is not None for c in tts.calls)

                    # Sólo texto, sin perfil.
                    tg.send_text(OWNER, "/texto")
                    await said("Sólo te mando los audios")
                    tg.send_text(OWNER, "/sinperfil")
                    await said("sin perfil")
                    tg.send_voice(OWNER, note)
                    await tg.wait_for(lambda: len(tg.sent("sendVoice")) == 6)
                    assert [v["caption"] for v in tg.sent("sendVoice")[4:]] == ["Inglés", "Francés"]

                    tg.send_text(OWNER, "hola")
                    await said("Mándame una nota de voz")
                finally:
                    await app.updater.stop()
                    await app.stop()

        asyncio.run(scenario())

    saved = json.loads((data_dir / "bot.json").read_text(encoding="utf-8"))
    assert saved[str(OWNER)]["targets"] == ["en", "fr"]
    assert saved[str(OWNER)]["accent"] == 0.1
    assert str(STRANGER) not in saved
    assert (data_dir / "perfiles" / "yo.wav").exists()


def test_settings_store_defaults(data_dir):
    store = SettingsStore(data_dir / "bot.json", ["de"])
    assert store.get(1).targets == ["de"]
    store.get(1).intensity = 0.8
    store.save()
    again = SettingsStore(data_dir / "bot.json")
    assert again.get(1).intensity == 0.8 and again.get(1).targets == ["de"]
