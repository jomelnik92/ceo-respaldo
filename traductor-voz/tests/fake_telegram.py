"""Un servidor de la API de bots de Telegram, falso y local, para probar el bot sin internet."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

BOT_ID = 999


def _user(user_id: int, is_bot: bool = False) -> dict:
    return {"id": user_id, "is_bot": is_bot, "first_name": f"Usuario {user_id}"}


class FakeTelegram:
    def __init__(self, token: str = "123456:FALSO"):
        self.token = token
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict]] = []
        self._updates: list[dict] = []
        self._next_update = 1
        self._next_message = 1000
        self._lock = threading.Lock()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        app = Starlette(
            routes=[
                Route("/bot{token}/{method}", self._api, methods=["GET", "POST"]),
                Route("/file/bot{token}/{path:path}", self._file, methods=["GET"]),
            ]
        )
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> FakeTelegram:
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("el servidor falso de Telegram no arrancó")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    # --- Lo que "hace el usuario" en Telegram -------------------------------------------------

    def send_text(self, user_id: int, text: str) -> None:
        message = self._message(user_id, text=text, **{"from": _user(user_id)})
        if text.startswith("/"):
            message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
        self._push(message)

    def send_voice(self, user_id: int, data: bytes, duration: int = 4) -> None:
        file_id = f"voz{len(self.files)}"
        self.files[file_id] = data
        voice = {"file_id": file_id, "file_unique_id": file_id, "duration": duration, "mime_type": "audio/ogg"}
        self._push(self._message(user_id, voice=voice, **{"from": _user(user_id)}))

    # --- Lo que "hizo el bot" ----------------------------------------------------------------

    def sent(self, method: str) -> list[dict]:
        return [params for m, params in list(self.calls) if m == method]

    def texts(self) -> list[str]:
        return [p["text"] for p in self.sent("sendMessage")]

    async def wait_for(self, predicate, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while not predicate():
            if time.time() > deadline:
                raise AssertionError(f"tiempo agotado; llamadas: {[m for m, _ in self.calls]}")
            await asyncio.sleep(0.05)

    # --- Servidor ----------------------------------------------------------------------------

    def _message(self, chat_id: int, **extra) -> dict:
        with self._lock:
            self._next_message += 1
            message_id = self._next_message
        return {"message_id": message_id, "date": int(time.time()), "chat": {"id": chat_id, "type": "private"}, **extra}

    def _push(self, message: dict) -> None:
        with self._lock:
            self._updates.append({"update_id": self._next_update, "message": message})
            self._next_update += 1

    async def _api(self, request):
        method = request.path_params["method"]
        params = {}
        for key, value in (await request.form()).multi_items():
            params[key] = await value.read() if hasattr(value, "read") else value
        self.calls.append((method, params))
        return JSONResponse({"ok": True, "result": await self._result(method, params)})

    async def _result(self, method: str, params: dict):
        chat_id = int(params.get("chat_id", 0))
        bot = _user(BOT_ID, is_bot=True)
        if method == "getMe":
            return {**bot, "username": "traductor_prueba_bot"}
        if method == "getUpdates":
            offset = int(params.get("offset", 0) or 0)
            deadline = time.time() + 0.3
            while True:
                with self._lock:
                    self._updates = [u for u in self._updates if u["update_id"] >= offset]
                    pending = list(self._updates)
                if pending or time.time() > deadline:
                    return pending
                await asyncio.sleep(0.02)
        if method == "getFile":
            file_id = params["file_id"]
            return {
                "file_id": file_id,
                "file_unique_id": file_id,
                "file_size": len(self.files[file_id]),
                "file_path": f"voice/{file_id}.oga",
            }
        if method == "sendMessage":
            return self._message(chat_id, text=params["text"], **{"from": bot})
        if method == "editMessageText":
            return {**self._message(chat_id, text=params["text"], **{"from": bot}), "message_id": int(params["message_id"])}
        if method == "sendVoice":
            voice = {"file_id": "salida", "file_unique_id": "salida", "duration": int(params.get("duration", 1))}
            return self._message(chat_id, voice=voice, caption=params.get("caption", ""), **{"from": bot})
        return True

    async def _file(self, request):
        file_id = request.path_params["path"].rsplit("/", 1)[-1].split(".")[0]
        return Response(self.files[file_id], media_type="audio/ogg")
