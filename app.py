import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


from websockets.legacy.client import connect as websocket_connect

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

LISTEN_MODEL = os.getenv("DEEPGRAM_LISTEN_MODEL", "nova-3")
LISTEN_VERSION = os.getenv("DEEPGRAM_LISTEN_VERSION", "v1").strip().lower()
LISTEN_LANGUAGE = os.getenv("DEEPGRAM_LISTEN_LANGUAGE", "").strip()
LISTEN_LANGUAGE_HINTS = [
    hint.strip()
    for hint in os.getenv("DEEPGRAM_LISTEN_LANGUAGE_HINTS", "").split(",")
    if hint.strip()
]
THINK_MODEL = os.getenv("DEEPGRAM_THINK_MODEL", "gemini-1.5-flash")
SPEAK_MODEL = os.getenv("DEEPGRAM_SPEAK_MODEL", "aura-2-thalia-en")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram").strip().lower()
ELEVEN_LABS_MODEL_ID = os.getenv("ELEVEN_LABS_MODEL_ID", "eleven_turbo_v2_5")
ELEVEN_LABS_LANGUAGE = os.getenv("ELEVEN_LABS_LANGUAGE", "").strip()
AGENT_LANGUAGE = os.getenv("DEEPGRAM_AGENT_LANGUAGE", "").strip()

# Map Deepgram short language codes to BCP-47 language codes required by ElevenLabs.
_LANG_CODE_MAP: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh",
    "ar": "ar",
    "ru": "ru",
    "nl": "nl",
    "pl": "pl",
    "sv": "sv",
    "tr": "tr",
}


def _elevenlabs_language_code(agent_language: str) -> str:
    """Return the BCP-47 language code ElevenLabs expects for the given agent language."""
    base = agent_language.split("-")[0].lower()
    return _LANG_CODE_MAP.get(base, base)


AGENT_GREETING = os.getenv("DEEPGRAM_AGENT_GREETING")
AGENT_PROMPT = os.getenv(
    "DEEPGRAM_AGENT_PROMPT",
    (
        "You are a concise, helpful voice assistant. Keep replies natural for "
        "spoken conversation."
    ),
)

app = FastAPI(title="Deepgram FastAPI Voice Agent")

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers.update(NO_CACHE_HEADERS)
        return response


app.mount("/assets", NoCacheStaticFiles(directory=FRONTEND), name="assets")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html", headers=NO_CACHE_HEADERS)


@app.get("/app.js")
async def legacy_app_js() -> FileResponse:
    return FileResponse(FRONTEND / "app.js", headers=NO_CACHE_HEADERS)


@app.get("/styles.css")
async def legacy_styles() -> FileResponse:
    return FileResponse(FRONTEND / "styles.css", headers=NO_CACHE_HEADERS)


@app.get("/pcm-worklet.js")
async def legacy_pcm_worklet() -> FileResponse:
    return FileResponse(FRONTEND / "pcm-worklet.js", headers=NO_CACHE_HEADERS)


@app.websocket("/ws/agent")
async def agent_proxy(client_ws: WebSocket) -> None:
    await client_ws.accept()

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        await client_ws.send_json(
            {
                "type": "Error",
                "message": "DEEPGRAM_API_KEY is not set on the backend.",
            }
        )
        await client_ws.close(code=1011)
        return

    headers = {"Authorization": f"Token {api_key}"}
    try:
        async with websocket_connect(
            DEEPGRAM_AGENT_URL,
            extra_headers=headers,
        ) as deepgram_ws:
            await run_proxy(client_ws, deepgram_ws)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        print("AGENT PROXY EXCEPTION:", exc)
        await safe_send_json(
            client_ws,
            {"type": "Error", "message": f"Agent proxy failed: {exc}"},
        )
        await safe_close(client_ws)


async def run_proxy(client_ws: WebSocket, deepgram_ws: Any) -> None:
    ready_to_stream = asyncio.Event()

    async def browser_to_deepgram() -> None:
        while True:
            message = await client_ws.receive()

            if "bytes" in message and message["bytes"] is not None:
                await ready_to_stream.wait()
                await deepgram_ws.send(message["bytes"])
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                if payload.get("type") == "Settings":
                    await deepgram_ws.send(json.dumps(payload))
                elif payload.get("type") == "KeepAlive":
                    await deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
                elif payload.get("type") == "InjectUserMessage":
                    await ready_to_stream.wait()
                    await deepgram_ws.send(json.dumps(payload))
            elif message.get("type") == "websocket.disconnect":
                break

    async def deepgram_to_browser() -> None:
        async for message in deepgram_ws:
            if isinstance(message, bytes):
                await client_ws.send_bytes(message)
                continue

            payload = json.loads(message)
            print(f"Deepgram sent: {payload.get('type')}", flush=True)
            if payload.get("type") == "Error":
                print(f"Deepgram Error: {payload}", flush=True)
            
            await client_ws.send_json(payload)
            if payload.get("type") == "Welcome":
                settings = default_settings()
                print(f"Agent speak settings: {settings['agent']['speak']}", flush=True)
                await deepgram_ws.send(json.dumps(settings))
            elif payload.get("type") == "SettingsApplied":
                ready_to_stream.set()

    tasks = [
        asyncio.create_task(browser_to_deepgram()),
        asyncio.create_task(deepgram_to_browser()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        task.result()


def default_settings() -> dict[str, Any]:
    eleven_labs_key = os.getenv("ELEVEN_LABS_API_KEY")
    eleven_labs_voice_id = os.getenv("ELEVEN_LABS_VOICE_ID", "RABOvaPec1ymXz02oDQi")

    if TTS_PROVIDER == "eleven_labs":
        if not eleven_labs_key:
            raise RuntimeError("TTS_PROVIDER=eleven_labs requires ELEVEN_LABS_API_KEY.")
        el_lang = ELEVEN_LABS_LANGUAGE or _elevenlabs_language_code(AGENT_LANGUAGE or "en")
        voice_id = quote(eleven_labs_voice_id, safe="")
        el_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            "/stream-input"
        )
        speak = {
            "provider": {
                "type": "eleven_labs",
                "model_id": ELEVEN_LABS_MODEL_ID,
                "language": el_lang,
            },
            "endpoint": {
                "url": el_url,
                "headers": {"xi-api-key": eleven_labs_key},
            },
        }
    elif TTS_PROVIDER == "deepgram":
        speak = {
            "provider": {
                "type": "deepgram",
                "model": SPEAK_MODEL,
            }
        }
    else:
        raise RuntimeError("TTS_PROVIDER must be either 'deepgram' or 'eleven_labs'.")

    if THINK_MODEL.startswith("gemini-"):
        think_provider = {
            "type": "google",
            "model": THINK_MODEL,
        }
    else:
        think_provider = {
            "type": "open_ai",
            "model": THINK_MODEL,
        }

    listen_provider: dict[str, Any] = {
        "type": "deepgram",
        "model": LISTEN_MODEL,
    }
    if LISTEN_VERSION == "v2":
        listen_provider["version"] = "v2"
    else:
        listen_provider["smart_format"] = True
        
    if LISTEN_LANGUAGE:
        listen_provider["language"] = LISTEN_LANGUAGE
    if LISTEN_LANGUAGE_HINTS:
        listen_provider["language_hints"] = LISTEN_LANGUAGE_HINTS

    agent_kwargs: dict[str, Any] = {
        "listen": {"provider": listen_provider},
        "think": {
            "provider": think_provider,
            "prompt": AGENT_PROMPT,
        },
        "speak": speak,
    }
    if AGENT_LANGUAGE:
        agent_kwargs["language"] = AGENT_LANGUAGE
    if AGENT_GREETING:
        agent_kwargs["greeting"] = AGENT_GREETING

    return {
        "type": "Settings",
        "experimental": False,
        "mip_opt_out": False,
        "flags": {"history": False},
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": 16000,
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none",
            },
        },
        "agent": agent_kwargs,
    }


async def safe_send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


async def safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080)
