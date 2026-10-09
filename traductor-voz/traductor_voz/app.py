"""App web (Gradio): graba o sube un audio y escúchalo en francés, alemán o inglés con tu voz."""

from __future__ import annotations

import time

import gradio as gr
import numpy as np

from . import languages
from .pipeline import Options, Result, Transcript, VoiceTranslator
from .profiles import list_profiles, load_profile, save_profile

NO_PROFILE = "(sin perfil: usar el mismo audio)"
DEFAULT_TARGETS = ["fr", "de", "en"]
TABLE_HEADERS = ["#", "Original", *languages.TARGETS.values()]
ANALYSIS_HEADERS = ["Idioma", "#", "Emoción", "Activación", "Intensidad", "Velocidad", "Tono (semitonos)"]

INTRO = """# Traductor de voz
Graba o sube un audio. Lo traduce y lo dice **con tu voz**, copiando el tono, el ritmo y la emoción
de cada frase. Si guardas un *perfil de voz* (pestaña de al lado) el timbre sale más limpio.
"""
PROFILE_HELP = """Lee en voz alta, con tu tono normal, entre **10 y 20 segundos** de texto,
en un lugar sin ruido ni eco. Esta grabación se usa para el timbre de tu voz;
la emoción se sigue tomando de cada audio que traduzcas."""


def _to_audio(res: Result) -> tuple[int, np.ndarray]:
    return res.sr, (np.clip(res.wav, -1, 1) * 32767).astype(np.int16)


def _table(tr: Transcript, texts: dict[str, list[str]]) -> list[list]:
    rows = []
    for i, phrase in enumerate(tr.phrases):
        rows.append([i + 1, phrase.text, *(texts.get(code, [""] * len(tr.phrases))[i] for code in languages.TARGETS)])
    return rows


def _analysis(results: dict[str, Result]) -> list[list]:
    rows = []
    for code, res in results.items():
        for r in res.report:
            rows.append(
                [
                    languages.name(code),
                    r.index,
                    r.emotion,
                    round(r.arousal, 2),
                    round(r.exaggeration, 2),
                    f"x{r.tempo:.2f}",
                    round(r.semitones, 1),
                ]
            )
    return rows


def _texts_from_table(table, n: int) -> dict[str, list[str]]:
    """Lee las traducciones (posiblemente corregidas a mano) de la tabla."""
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    texts: dict[str, list[str]] = {}
    for j, code in enumerate(languages.TARGETS, start=2):
        column = [str(row[j]) if j < len(row) and row[j] is not None else "" for row in rows]
        if len(column) == n and any(c.strip() for c in column):
            texts[code] = column
    return texts


def build_ui(vt: VoiceTranslator) -> gr.Blocks:
    def options(intensity, accent, pitch, timing, pauses) -> Options:
        return Options(
            intensity=intensity, accent=accent, pitch_match=pitch, timing_match=timing, keep_pauses=pauses
        )

    def profile_ref(name):
        return None if not name or name == NO_PROFILE else load_profile(name)

    def render(tr: Transcript, texts: dict[str, list[str]], results: dict[str, Result], elapsed: float):
        audios = [
            gr.Audio(value=_to_audio(results[code]) if code in results else None, visible=code in results)
            for code in languages.TARGETS
        ]
        status = (
            f"Idioma del audio: **{languages.name(tr.language)}** · {len(tr.phrases)} frases · "
            f"listo en {elapsed:.0f} s"
        )
        return [tr, *audios, _table(tr, texts), _analysis(results), status]

    def synthesize_all(tr, targets, texts, profile, opts, progress):
        results = {}
        for k, code in enumerate(targets):
            if code not in texts:
                progress(k / len(targets), desc=f"Traduciendo al {languages.name(code).lower()}")
                texts[code] = vt.translate(tr, code)

            def step(frac, desc, k=k, code=code):
                progress((k + frac) / len(targets), desc=f"{languages.name(code)}: {desc}")

            results[code] = vt.synthesize(tr, code, texts[code], opts, profile, step)
        return results

    def on_translate(audio, targets, profile, intensity, accent, pitch, timing, pauses, source, progress=gr.Progress()):
        if not audio:
            raise gr.Error("Primero graba o sube un audio.")
        if not targets:
            raise gr.Error("Elige al menos un idioma.")
        t0 = time.time()
        try:
            progress(0, desc="Transcribiendo")
            tr = vt.transcribe(audio, None if source == "auto" else source)
            texts: dict[str, list[str]] = {}
            results = synthesize_all(
                tr, targets, texts, profile_ref(profile), options(intensity, accent, pitch, timing, pauses), progress
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise gr.Error(str(e)) from e
        return render(tr, texts, results, time.time() - t0)

    def on_regenerate(tr, table, targets, profile, intensity, accent, pitch, timing, pauses, progress=gr.Progress()):
        if tr is None:
            raise gr.Error("Primero traduce un audio.")
        if not targets:
            raise gr.Error("Elige al menos un idioma.")
        t0 = time.time()
        texts = _texts_from_table(table, len(tr.phrases))
        try:
            results = synthesize_all(
                tr, targets, texts, profile_ref(profile), options(intensity, accent, pitch, timing, pauses), progress
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise gr.Error(str(e)) from e
        return render(tr, texts, results, time.time() - t0)

    def on_save_profile(audio, name):
        if not audio:
            raise gr.Error("Primero graba o sube tu lectura.")
        try:
            path = save_profile(name, audio, vt.asr)
        except (ValueError, RuntimeError) as e:
            raise gr.Error(str(e)) from e
        return (
            f"Perfil **{path.stem}** guardado.",
            gr.Dropdown(choices=[NO_PROFILE, *list_profiles()], value=path.stem),
        )

    with gr.Blocks(title="Traductor de voz") as demo:
        transcript = gr.State(None)
        gr.Markdown(INTRO)
        with gr.Tab("Traducir"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tu audio")
                    targets = gr.CheckboxGroup(
                        [(name, code) for code, name in languages.TARGETS.items()],
                        value=DEFAULT_TARGETS,
                        label="Traducir a",
                    )
                    profile = gr.Dropdown([NO_PROFILE, *list_profiles()], value=NO_PROFILE, label="Perfil de voz")
                    with gr.Accordion("Ajustes de voz", open=False):
                        intensity = gr.Slider(0.25, 1.0, 0.5, step=0.05, label="Intensidad emocional")
                        accent = gr.Slider(0.0, 0.6, 0.3, step=0.05, label="Acento: 0 = nativo · 0.6 = el tuyo")
                        pitch = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="Copiar tu tono en cada frase")
                        timing = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="Copiar tu ritmo en cada frase")
                        pauses = gr.Checkbox(True, label="Respetar mis pausas")
                        source = gr.Dropdown(
                            [("Detectar automáticamente", "auto"), *((n, c) for c, n in languages.SOURCES.items())],
                            value="auto",
                            label="Idioma del audio",
                        )
                    go = gr.Button("Traducir", variant="primary")
                with gr.Column():
                    outs = [
                        gr.Audio(label=name, type="numpy", interactive=False, visible=False)
                        for name in languages.TARGETS.values()
                    ]
                    status = gr.Markdown()
            with gr.Accordion("Textos: corrige la traducción y regenera", open=False):
                table = gr.Dataframe(headers=TABLE_HEADERS, type="array", interactive=True, wrap=True)
                regen = gr.Button("Regenerar voz con estos textos")
            with gr.Accordion("Emoción detectada en cada frase", open=False):
                analysis = gr.Dataframe(headers=ANALYSIS_HEADERS, type="array", interactive=False)

        with gr.Tab("Mi perfil de voz"):
            gr.Markdown(PROFILE_HELP)
            p_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tu lectura")
            p_name = gr.Textbox(value="yo", label="Nombre del perfil")
            p_save = gr.Button("Guardar perfil", variant="primary")
            p_status = gr.Markdown()

        sliders = [intensity, accent, pitch, timing, pauses]
        results_out = [transcript, *outs, table, analysis, status]
        go.click(
            on_translate,
            [audio_in, targets, profile, *sliders, source],
            results_out,
            concurrency_limit=1,
            concurrency_id="modelos",
        )
        regen.click(
            on_regenerate,
            [transcript, table, targets, profile, *sliders],
            results_out,
            concurrency_limit=1,
            concurrency_id="modelos",
        )
        p_save.click(on_save_profile, [p_audio, p_name], [p_status, profile], concurrency_id="modelos")
    return demo


def launch(
    tts: str = "chatterbox",
    whisper_model: str | None = None,
    translation_engine: str = "auto",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    preload: bool = True,
) -> None:
    vt = VoiceTranslator(tts=tts, whisper_model=whisper_model, translation_engine=translation_engine)
    if preload:
        print("Cargando modelos (la primera vez se descargan, puede tardar)...")
        vt.asr  # noqa: B018 - fuerza la carga
        vt.tts  # noqa: B018
    build_ui(vt).queue().launch(server_name=host, server_port=port, share=share)
