# CLAUDE.md

Este repositorio tiene dos cosas independientes:

- `index.html`, `privacidad.html`: página pública de CEO respaldo (herramienta personal de respaldo
  para Windows). No tocarlas salvo que se pida.
- `traductor-voz/`: app que traduce audios a francés, alemán e inglés con la voz clonada del
  dueño, conservando tono, ritmo y emoción. Uso e instalación: `traductor-voz/README.md`.

Se habla en español con el dueño. En el código: identificadores en inglés; comentarios, textos de
la interfaz y mensajes al usuario en español.

## Traductor de voz: estado

Escrito en una sesión en la nube donde Hugging Face y Telegram estaban bloqueados, así que:

- **Probado**: 41 pruebas con motores simulados y voces sintéticas (Python 3.11 y 3.12); el motor
  Chatterbox con su código real pero pesos aleatorios; el bot con la librería real de Telegram
  contra un servidor de Telegram falso.
- **Nunca corrido**: los modelos reales (Whisper, MarianMT/NLLB, Chatterbox, Qwen3-TTS), la
  calidad de la voz, el bot contra Telegram real, y nada en Windows.

PC del dueño: Windows, NVIDIA RTX 3090 (24 GB), 64 GB de RAM, Intel Core i9 de 14ª generación.

## Próximos pasos

1. Instalar en Windows según el README (Python 3.11, ffmpeg, torch 2.6.0 cu124,
   `pip install -e ".[chatterbox,bot,dev]"`) y correr `pytest`.
2. Primera prueba real con un audio del dueño: `traductor-voz traducir nota.ogg -i en`, y escuchar.
3. Ajustar con audios reales: umbrales de `prosody.arousal` y `prosody.exaggeration`, valores por
   defecto de `pipeline.Options` (accent → cfg_weight, timing_match, pitch_match), calidad de la
   traducción.
4. Bot: crear el bot con @BotFather y probarlo contra Telegram real.
5. Comparar Chatterbox con Qwen3-TTS (cada uno en su propio entorno).

## Comandos

```
cd traductor-voz
pytest                                   # no descarga modelos
traductor-voz app                        # app web en http://127.0.0.1:7860
traductor-voz bot                        # usa TELEGRAM_TOKEN y TELEGRAM_USUARIOS
traductor-voz traducir AUDIO -i fr de en
traductor-voz perfil crear yo lectura.wav
```

## Arquitectura (`traductor-voz/traductor_voz/`)

- `pipeline.py`: `VoiceTranslator` → `transcribe` (Whisper + prosodia por frase), `translate`,
  `synthesize` (voz por frase, `_match_prosody` corrige tono, ritmo y volumen, y respeta pausas).
- `prosody.py`: F0 con pyin, nivel, variación de entonación → activación → `exaggeration`.
- `segments.py`: une segmentos de Whisper en frases, ventana de referencia por frase, parte textos largos.
- `audio.py`: lectura/escritura con ffmpeg, tempo y tono (rubberband, o asetrate+atempo), nivel.
- `engines/`: `asr.py` (faster-whisper), `translate.py` (MarianMT, NLLB-200),
  `chatterbox_engine.py` (timbre del perfil + prompt de tokens de la frase original),
  `qwen_engine.py` (clonación por contexto con audio + texto de la frase).
- `app.py` (Gradio), `bot.py` (python-telegram-bot; los modelos corren en un solo hilo),
  `cli.py`, `profiles.py` (datos en `~/.traductor-voz`, o `TRADUCTOR_VOZ_DATOS`).

## Trampas conocidas

- Chatterbox fija torch==2.6.0, transformers==5.2.0 y gradio==6.8.0; Qwen3-TTS fija
  transformers==4.57.3: un entorno virtual por motor.
- resemble-perth (marca de agua de Chatterbox) necesita `pkg_resources`: `setuptools<81`
  (ya incluido en el extra `chatterbox`). Sin eso falla con `'NoneType' object is not callable`.
- Chatterbox: cuando la referencia está en otro idioma, un `cfg_weight` bajo (`Options.accent`)
  reduce el acento heredado.
- Las RTX serie 50 necesitan torch ≥ 2.7, que choca con el pin de Chatterbox (la 3090 no tiene ese problema).
