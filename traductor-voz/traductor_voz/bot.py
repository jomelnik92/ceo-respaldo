"""Bot de Telegram: le mandas una nota de voz y te contesta con notas de voz tuyas en otros idiomas.

El bot corre en tu computadora (los modelos usan tu GPU). Telegram sólo transporta los audios.
Sólo atiende a los usuarios autorizados; a cualquier otro le contesta con su número de usuario
para que puedas autorizarlo si quieres.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from . import languages
from .audio import save_audio
from .pipeline import Options, VoiceTranslator
from .profiles import data_dir, list_profiles, load_profile, save_profile

log = logging.getLogger(__name__)

MAX_AUDIO_SEC = 300
MAX_CAPTION = 1024
MAX_MESSAGE = 4000
AUDIO_FILTER = filters.VOICE | filters.AUDIO | filters.Document.AUDIO | filters.VIDEO_NOTE | filters.VIDEO

COMMANDS = [
    ("idiomas", "elegir idiomas, ej.: /idiomas fr de en"),
    ("perfil", "ver tu perfil de voz"),
    ("grabarperfil", "grabar un perfil de voz nuevo"),
    ("usarperfil", "usar un perfil guardado"),
    ("sinperfil", "dejar de usar perfil"),
    ("intensidad", "intensidad emocional, 0.25 a 1"),
    ("acento", "0 = nativo, 0.6 = el tuyo"),
    ("texto", "mostrar u ocultar el texto traducido"),
    ("ayuda", "cómo se usa"),
]


@dataclass
class ChatSettings:
    targets: list[str] = field(default_factory=lambda: ["fr", "de", "en"])
    profile: str | None = None
    intensity: float = 0.5
    accent: float = 0.3
    show_text: bool = True


class SettingsStore:
    """Ajustes de cada chat, guardados en un json para que sobrevivan a reinicios."""

    def __init__(self, path: Path, default_targets: list[str] | None = None):
        self.path = path
        self.default_targets = default_targets
        self._data: dict[str, ChatSettings] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._data = {k: ChatSettings(**v) for k, v in raw.items()}

    def get(self, chat_id: int) -> ChatSettings:
        key = str(chat_id)
        if key not in self._data:
            self._data[key] = ChatSettings()
            if self.default_targets:
                self._data[key].targets = list(self.default_targets)
        return self._data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._data.items()}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _seconds(value) -> float | None:
    """Telegram da las duraciones como int o como timedelta según la versión de la librería."""
    if value is None:
        return None
    return value.total_seconds() if hasattr(value, "total_seconds") else float(value)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TranslatorBot:
    def __init__(self, vt: VoiceTranslator, allowed_users: set[int], store: SettingsStore):
        self.vt = vt
        self.allowed_users = allowed_users
        self.store = store
        # Un solo hilo para los modelos: los trabajos se hacen de a uno y la GPU no se satura.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelos")
        self.waiting_profile: dict[int, str] = {}

    async def _run(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.executor, partial(fn, *args, **kwargs))

    def build(self, token: str, base_url: str | None = None, base_file_url: str | None = None) -> Application:
        builder = Application.builder().token(token).concurrent_updates(True).post_init(self._post_init)
        if base_url:
            builder = builder.base_url(base_url).base_file_url(base_file_url or base_url)
        app = builder.build()
        allowed = filters.User(user_id=self.allowed_users)
        new_message = filters.UpdateType.MESSAGE
        handlers = {
            "start": self.cmd_help,
            "ayuda": self.cmd_help,
            "idiomas": self.cmd_languages,
            "perfil": self.cmd_profile,
            "grabarperfil": self.cmd_record_profile,
            "usarperfil": self.cmd_use_profile,
            "sinperfil": self.cmd_no_profile,
            "cancelar": self.cmd_cancel,
            "intensidad": self.cmd_intensity,
            "acento": self.cmd_accent,
            "texto": self.cmd_text,
        }
        for name, fn in handlers.items():
            app.add_handler(CommandHandler(name, fn, filters=allowed))
        app.add_handler(MessageHandler(allowed & new_message & AUDIO_FILTER, self.on_audio))
        app.add_handler(MessageHandler(allowed & new_message & filters.TEXT & ~filters.COMMAND, self.on_text))
        app.add_handler(MessageHandler(~allowed & new_message, self.on_stranger))
        app.add_error_handler(self.on_error)
        return app

    async def _post_init(self, app: Application) -> None:
        await app.bot.set_my_commands([BotCommand(c, d) for c, d in COMMANDS])

    # --- Comandos ---------------------------------------------------------------------------

    def _summary(self, s: ChatSettings) -> str:
        targets = ", ".join(languages.name(c) for c in s.targets)
        return (
            f"Idiomas: {targets}\n"
            f"Perfil de voz: {s.profile or 'ninguno (uso el mismo audio)'}\n"
            f"Intensidad emocional: {s.intensity:g} · Acento: {s.accent:g} · "
            f"Texto: {'sí' if s.show_text else 'no'}"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.store.get(update.effective_chat.id)
        await update.effective_message.reply_text(
            "Mándame una nota de voz y te la devuelvo con tu voz en otros idiomas.\n\n"
            + self._summary(s)
            + "\n\nComandos:\n"
            + "\n".join(f"/{c} – {d}" for c, d in COMMANDS)
        )

    async def cmd_languages(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.store.get(update.effective_chat.id)
        codes = [a.lower() for a in context.args]
        options = ", ".join(f"{c} ({n})" for c, n in languages.TARGETS.items())
        if not codes:
            await update.effective_message.reply_text(
                f"Ahora: {', '.join(s.targets)}.\nOpciones: {options}\nEjemplo: /idiomas fr en"
            )
            return
        unknown = [c for c in codes if c not in languages.TARGETS]
        if unknown:
            await update.effective_message.reply_text(f"No conozco: {', '.join(unknown)}. Opciones: {options}")
            return
        s.targets = list(dict.fromkeys(codes))
        self.store.save()
        await update.effective_message.reply_text(
            "Idiomas: " + ", ".join(languages.name(c) for c in s.targets)
        )

    async def cmd_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.store.get(update.effective_chat.id)
        saved = ", ".join(list_profiles()) or "ninguno"
        await update.effective_message.reply_text(
            f"Perfil en uso: {s.profile or 'ninguno'}\nGuardados: {saved}\n\n"
            "/grabarperfil [nombre] para grabar uno nuevo\n/usarperfil nombre para usar uno guardado"
        )

    async def cmd_record_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        name = " ".join(context.args).strip() or "yo"
        self.waiting_profile[update.effective_chat.id] = name
        await update.effective_message.reply_text(
            f"Mándame ahora una nota de voz de 10 a 20 segundos para el perfil «{name}»: "
            "lee algo con tu tono normal, en un lugar sin ruido. (/cancelar para salir)"
        )

    async def cmd_use_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        name = " ".join(context.args).strip()
        if name not in list_profiles():
            await update.effective_message.reply_text(
                f"No existe el perfil «{name}». Guardados: {', '.join(list_profiles()) or 'ninguno'}"
            )
            return
        s = self.store.get(update.effective_chat.id)
        s.profile = name
        self.store.save()
        await update.effective_message.reply_text(f"Usando el perfil «{name}».")

    async def cmd_no_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.store.get(update.effective_chat.id)
        s.profile = None
        self.store.save()
        await update.effective_message.reply_text("Listo: sin perfil, uso el timbre de cada audio.")

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.waiting_profile.pop(update.effective_chat.id, None)
        await update.effective_message.reply_text("Cancelado.")

    async def _set_number(self, update, context, attr: str, label: str, lo: float, hi: float) -> None:
        s = self.store.get(update.effective_chat.id)
        try:
            value = float(context.args[0].replace(",", "."))
            if not lo <= value <= hi:
                raise ValueError
        except (IndexError, ValueError):
            await update.effective_message.reply_text(
                f"{label}: ahora {getattr(s, attr):g}. Usa un número entre {lo:g} y {hi:g}."
            )
            return
        setattr(s, attr, value)
        self.store.save()
        await update.effective_message.reply_text(f"{label}: {value:g}")

    async def cmd_intensity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_number(update, context, "intensity", "Intensidad emocional", 0.25, 1.0)

    async def cmd_accent(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_number(update, context, "accent", "Acento", 0.0, 0.6)

    async def cmd_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.store.get(update.effective_chat.id)
        s.show_text = not s.show_text
        self.store.save()
        await update.effective_message.reply_text(
            "Te mando el texto traducido con cada audio." if s.show_text else "Sólo te mando los audios."
        )

    # --- Mensajes ---------------------------------------------------------------------------

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text("Mándame una nota de voz. /ayuda para ver las opciones.")

    async def on_stranger(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if update.effective_message is None or user is None:
            return
        log.warning("Usuario no autorizado: %s (%s)", user.id, user.full_name)
        await update.effective_message.reply_text(
            f"Este bot es privado. Tu número de usuario de Telegram es {user.id}.\n"
            "Si es tu bot, agrégalo con --usuarios o TELEGRAM_USUARIOS y reinícialo."
        )

    async def on_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat_id = update.effective_chat.id
        media = msg.voice or msg.audio or msg.document or msg.video_note or msg.video
        duration = _seconds(getattr(media, "duration", None))
        if duration is not None and duration > MAX_AUDIO_SEC:
            await msg.reply_text(f"El audio es muy largo: el máximo es {MAX_AUDIO_SEC // 60} minutos.")
            return

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "entrada"
            file = await media.get_file()
            await file.download_to_drive(src)

            if chat_id in self.waiting_profile:
                await self._save_profile(update, src)
                return

            s = self.store.get(chat_id)
            status = await msg.reply_text("Recibido, escuchando…")
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            try:
                tr = await self._run(self.vt.transcribe, src)
            except ValueError as e:
                await status.edit_text(str(e))
                return
            original = " ".join(p.text for p in tr.phrases)
            targets = [c for c in s.targets if c != tr.language]
            pending = ", ".join(languages.name(c) for c in targets) or "nada (ya está en ese idioma)"
            await status.edit_text(
                _clip(f"Entendí ({languages.name(tr.language)}): «{original}»\n\nPreparando: {pending}…", MAX_MESSAGE)
            )

            profile = None
            if s.profile:
                try:
                    profile = await self._run(load_profile, s.profile)
                except FileNotFoundError:
                    await msg.reply_text(f"No encontré el perfil «{s.profile}»; sigo sin perfil.")
                    s.profile = None
                    self.store.save()

            opts = Options(intensity=s.intensity, accent=s.accent)
            for code in targets:
                await context.bot.send_chat_action(chat_id, ChatAction.RECORD_VOICE)
                texts = await self._run(self.vt.translate, tr, code)
                res = await self._run(self.vt.synthesize, tr, code, texts, opts, profile)
                out = await self._run(save_audio, Path(tmp) / f"{code}.ogg", res.wav, res.sr)
                caption = languages.name(code)
                if s.show_text:
                    caption = _clip(f"{caption}: {' '.join(t for t in texts if t.strip())}", MAX_CAPTION)
                with open(out, "rb") as fh:
                    await msg.reply_voice(voice=fh, caption=caption, duration=round(len(res.wav) / res.sr))

    async def _save_profile(self, update: Update, src: Path) -> None:
        chat_id = update.effective_chat.id
        name = self.waiting_profile[chat_id]
        try:
            path = await self._run(save_profile, name, src, self.vt.asr)
        except ValueError as e:
            await update.effective_message.reply_text(f"{e} Prueba de nuevo o /cancelar.")
            return
        del self.waiting_profile[chat_id]
        s = self.store.get(chat_id)
        s.profile = path.stem
        self.store.save()
        await update.effective_message.reply_text(
            f"Perfil «{path.stem}» guardado y en uso. Ahora mándame lo que quieras traducir."
        )

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Error procesando un mensaje", exc_info=context.error)
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(f"Algo salió mal: {context.error}")


def run_bot(
    token: str,
    allowed_users: set[int],
    tts: str = "chatterbox",
    whisper_model: str | None = None,
    translation_engine: str = "auto",
    default_targets: list[str] | None = None,
) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    vt = VoiceTranslator(tts=tts, whisper_model=whisper_model, translation_engine=translation_engine)
    print("Cargando modelos (la primera vez se descargan, puede tardar)...")
    vt.asr  # noqa: B018 - fuerza la carga
    vt.tts  # noqa: B018
    store = SettingsStore(data_dir() / "bot.json", default_targets)
    app = TranslatorBot(vt, allowed_users, store).build(token)
    if allowed_users:
        print(f"Bot listo. Usuarios autorizados: {', '.join(map(str, sorted(allowed_users)))}")
    else:
        print("Bot listo, pero SIN usuarios autorizados: escríbele y te dirá tu número de usuario.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
