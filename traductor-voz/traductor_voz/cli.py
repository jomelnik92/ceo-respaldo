"""Línea de comandos.

    traductor-voz app                                   # abre la app web
    traductor-voz bot --token XXX --usuarios 123        # bot de Telegram
    traductor-voz traducir nota.ogg -i fr de en         # traduce un audio
    traductor-voz perfil crear yo lectura.wav           # guarda tu perfil de voz
    traductor-voz perfil listar
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import languages
from .engines import TTS_ENGINES


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--motor", choices=TTS_ENGINES, default="chatterbox", help="motor de voz (por defecto: chatterbox)")
    p.add_argument("--whisper", default=None, help="modelo de Whisper (por defecto: large-v3-turbo en GPU, small en CPU)")
    p.add_argument("--traduccion", choices=("auto", "marian", "nllb"), default="auto", help="motor de traducción")


def _translator(args):
    from .pipeline import VoiceTranslator

    return VoiceTranslator(tts=args.motor, whisper_model=args.whisper, translation_engine=args.traduccion)


def cmd_translate(args) -> int:
    from .audio import save_audio
    from .pipeline import Options
    from .profiles import load_profile

    opts = Options(
        intensity=args.intensidad,
        accent=args.acento,
        pitch_match=args.tono,
        timing_match=args.ritmo,
        keep_pauses=not args.sin_pausas,
    )
    profile = load_profile(args.perfil) if args.perfil else None
    vt = _translator(args)
    print("Transcribiendo...", file=sys.stderr)
    tr = vt.transcribe(args.audio, args.origen)
    print(f"Idioma detectado: {languages.name(tr.language)} · {len(tr.phrases)} frases", file=sys.stderr)
    out_dir = Path(args.salida)
    for target in args.idiomas:
        print(f"→ {languages.name(target)}", file=sys.stderr)
        texts = vt.translate(tr, target)
        res = vt.synthesize(tr, target, texts, opts, profile)
        path = save_audio(out_dir / f"{Path(args.audio).stem}.{target}.{args.formato}", res.wav, res.sr)
        for r in res.report:
            print(f"  [{r.index}] ({r.emotion}, emoción {r.exaggeration:.2f}) {r.translation}", file=sys.stderr)
        print(path)
    return 0


def cmd_profile(args) -> int:
    from .profiles import list_profiles, save_profile

    if args.accion == "crear":
        asr = None
        if not args.sin_transcribir:
            from .engines.asr import WhisperASR

            asr = WhisperASR(args.whisper)
        print(save_profile(args.nombre, args.audio, asr))
    else:
        for name in list_profiles():
            print(name)
    return 0


def cmd_app(args) -> int:
    from .app import launch

    launch(
        tts=args.motor,
        whisper_model=args.whisper,
        translation_engine=args.traduccion,
        host=args.host,
        port=args.puerto,
        share=args.compartir,
    )
    return 0


def _parse_users(value: str) -> set[int]:
    try:
        return {int(u) for u in value.replace(";", ",").split(",") if u.strip()}
    except ValueError:
        raise ValueError(f"--usuarios debe ser una lista de números separados por comas, no {value!r}") from None


def cmd_bot(args) -> int:
    from .bot import run_bot

    if not args.token:
        raise ValueError("Falta el token del bot: usa --token o la variable TELEGRAM_TOKEN (te lo da @BotFather).")
    run_bot(
        token=args.token,
        allowed_users=_parse_users(args.usuarios),
        tts=args.motor,
        whisper_model=args.whisper,
        translation_engine=args.traduccion,
        default_targets=args.idiomas,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traductor-voz", description="Traduce tus audios con tu propia voz.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("app", help="abre la app web")
    _add_engine_args(p)
    p.add_argument("--host", default="127.0.0.1", help="usa 0.0.0.0 para abrirla desde el celular en tu red")
    p.add_argument("--puerto", type=int, default=7860)
    p.add_argument("--compartir", action="store_true", help="crea un enlace público temporal de Gradio")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("bot", help="bot de Telegram")
    p.add_argument("--token", default=os.environ.get("TELEGRAM_TOKEN"), help="token de @BotFather (o TELEGRAM_TOKEN)")
    p.add_argument(
        "--usuarios",
        default=os.environ.get("TELEGRAM_USUARIOS", ""),
        help="números de usuario autorizados, separados por comas (o TELEGRAM_USUARIOS)",
    )
    p.add_argument("-i", "--idiomas", nargs="+", default=["fr", "de", "en"], choices=list(languages.TARGETS))
    _add_engine_args(p)
    p.set_defaults(func=cmd_bot)

    p = sub.add_parser("traducir", help="traduce un archivo de audio")
    p.add_argument("audio", help="archivo de audio (wav, mp3, ogg, m4a...)")
    p.add_argument("-i", "--idiomas", nargs="+", default=["fr", "de", "en"], choices=list(languages.TARGETS))
    p.add_argument("-o", "--salida", default="salida", help="carpeta de salida")
    p.add_argument("-f", "--formato", default="wav", choices=("wav", "mp3", "ogg", "m4a"))
    p.add_argument("--origen", default=None, help="idioma del audio (por defecto lo detecta)")
    p.add_argument("--perfil", default=None, help="perfil de voz guardado")
    p.add_argument("--intensidad", type=float, default=0.5, help="intensidad emocional (0.25-1.0)")
    p.add_argument("--acento", type=float, default=0.3, help="0 = acento nativo, 0.5 = conserva tu acento")
    p.add_argument("--tono", type=float, default=1.0, help="cuánto copiar tu tono medio (0-1)")
    p.add_argument("--ritmo", type=float, default=0.5, help="cuánto copiar la duración de cada frase (0-1)")
    p.add_argument("--sin-pausas", action="store_true", help="no respetar tus pausas originales")
    _add_engine_args(p)
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("perfil", help="perfiles de voz")
    p.add_argument("accion", choices=("crear", "listar"))
    p.add_argument("nombre", nargs="?")
    p.add_argument("audio", nargs="?", help="grabación limpia de 10-20 s")
    p.add_argument("--whisper", default=None)
    p.add_argument("--sin-transcribir", action="store_true", help="no transcribir el perfil (Qwen3 clona peor sin texto)")
    p.set_defaults(func=cmd_profile)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "perfil" and args.accion == "crear" and not (args.nombre and args.audio):
        parser.error("uso: traductor-voz perfil crear NOMBRE AUDIO")
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
