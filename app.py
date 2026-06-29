import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
ELEVEN_LABS_MODEL_ID = os.getenv("ELEVEN_LABS_MODEL_ID", "eleven_flash_v2_5")
ELEVEN_LABS_LANGUAGE = os.getenv("ELEVEN_LABS_LANGUAGE", "").strip()
ELEVEN_LABS_OPTIMIZE_STREAMING_LATENCY = os.getenv(
    "ELEVEN_LABS_OPTIMIZE_STREAMING_LATENCY",
    "4",
).strip()
AGENT_LANGUAGE = os.getenv("DEEPGRAM_AGENT_LANGUAGE", "").strip()
ENABLE_ELEVEN_LABS_LANGUAGE_ROUTING = (
    os.getenv("ENABLE_ELEVEN_LABS_LANGUAGE_ROUTING", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
STRICT_NATIVE_INDIAN_VOICE_ROUTING = (
    os.getenv("STRICT_NATIVE_INDIAN_VOICE_ROUTING", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Map Deepgram language codes -> ElevenLabs BCP-47 language codes
_LANG_CODE_MAP: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "mr": "mr",
    "gu": "gu",
    "kn": "kn",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
    "ml": "ml",
    "pa": "pa",
    "or": "or",
    "as": "as",
    "ur": "ur",
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

_INDIAN_LANGUAGE_CODES = {
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "ur",
}

_ELEVEN_LABS_PROVIDER_LANGUAGE_MAP: dict[str, str] = {
    "hi": "hi",
    "mr": "hi",
    "gu": "hi",
    "kn": "hi",
    "pa": "hi",
    "ur": "ur",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
    "bn": "bn",
    "or": "or",
    "as": "as",
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "mr": "mr",
    "mar": "mr",
    "marathi": "mr",
    "gu": "gu",
    "guj": "gu",
    "gujarati": "gu",
    "kn": "kn",
    "kan": "kn",
    "kannada": "kn",
    "ta": "ta",
    "tam": "ta",
    "tamil": "ta",
    "te": "te",
    "tel": "te",
    "telugu": "te",
    "bn": "bn",
    "ben": "bn",
    "bengali": "bn",
    "bangla": "bn",
    "ml": "ml",
    "mal": "ml",
    "malayalam": "ml",
    "pa": "pa",
    "pan": "pa",
    "punjabi": "pa",
    "panjabi": "pa",
    "ur": "ur",
    "urd": "ur",
    "urdu": "ur",
    "or": "or",
    "ori": "or",
    "odia": "or",
    "oriya": "or",
    "as": "as",
    "asm": "as",
    "assamese": "as",
}

_INDIAN_VOICE_FALLBACK_LANGUAGE = "hi"
_DEVANAGARI_LANGUAGE_CODES = {"hi", "mr"}
_BENGALI_ASSAMESE_LANGUAGE_CODES = {"bn", "as"}
_COMMON_ENGLISH_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "good",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "is",
    "me",
    "morning",
    "my",
    "name",
    "please",
    "thanks",
    "thank",
    "the",
    "there",
    "what",
    "who",
    "why",
    "yes",
    "you",
    "your",
}


def _base_language_code(language: str) -> str:
    normalized = (language or "").strip().lower().replace("_", "-")
    if not normalized:
        return ""
    if normalized in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[normalized]

    base = normalized.split("-")[0]
    return _LANGUAGE_ALIASES.get(base, base)


def _env_voice_id(language: str, default: str) -> str:
    return os.getenv(f"ELEVEN_LABS_VOICE_ID_{language.upper()}", default).strip() or default


# Native-accent ElevenLabs voice IDs per base language code.
# These defaults are env-overridable so you can pin voice IDs that match your account.
_LANGUAGE_VOICE_MAP: dict[str, str] = {
    "en": _env_voice_id("en", "cgSgspJ2msm6clMCkdW9"),
    "hi": _env_voice_id("hi", "AQG24GkRyuUUm3KHVSaN"),
    "mr": _env_voice_id("mr", "aZaMD7e0iu6aJpSocGIY"),
    "gu": _env_voice_id("gu", "aVwNS3gNch2GhQaTeQ54"),
    "kn": _env_voice_id("kn", "7B4TkucyQHy3r9hvAnhg"),
    "ta": _env_voice_id("ta", "C2RGMrNBTZaNfddRPeRH"),
    "te": _env_voice_id("te", "QKyvRuehpb8zB3cRkzIn"),
    "bn": _env_voice_id("bn", "70QpbCWFDvTpWo8ZKOUb"),
    "ml": _env_voice_id("ml", "OVkoEbwxsYHiSRMFV9t3"),
    "pa": _env_voice_id("pa", "TMoHpmi3HTyjpoLh2KMT"),
    "ur": _env_voice_id("ur", "o85TqPN3F4P7dWae2paA"),
    "or": _env_voice_id("or", "AQG24GkRyuUUm3KHVSaN"),
    "as": _env_voice_id("as", "AQG24GkRyuUUm3KHVSaN"),
}


def _elevenlabs_language_code(agent_language: str) -> str:
    """Return the provider language code ElevenLabs expects for the configured voice family."""
    base = _base_language_code(agent_language)
    provider_lang = _ELEVEN_LABS_PROVIDER_LANGUAGE_MAP.get(base, base)
    if STRICT_NATIVE_INDIAN_VOICE_ROUTING and base in _INDIAN_LANGUAGE_CODES:
        provider_lang = _ELEVEN_LABS_PROVIDER_LANGUAGE_MAP.get(
            base,
            _INDIAN_VOICE_FALLBACK_LANGUAGE,
        )
    return _LANG_CODE_MAP.get(provider_lang, provider_lang)


def _elevenlabs_voice_id(agent_language: str) -> str:
    base = _base_language_code(agent_language)
    if STRICT_NATIVE_INDIAN_VOICE_ROUTING and base in _INDIAN_LANGUAGE_CODES:
        native_voice_id = _LANGUAGE_VOICE_MAP.get(base)
        if native_voice_id:
            return native_voice_id
        return _LANGUAGE_VOICE_MAP[_INDIAN_VOICE_FALLBACK_LANGUAGE]
    return _LANGUAGE_VOICE_MAP.get(base, _LANGUAGE_VOICE_MAP["en"])


def _resolved_elevenlabs_tts_language(agent_language: str) -> str:
    """Honor an explicit multilingual TTS setting instead of constraining output per turn."""
    configured = ELEVEN_LABS_LANGUAGE.strip().lower()
    if configured:
        return configured
    return _elevenlabs_language_code(agent_language)


def _contains_codepoint(text: str, start: int, end: int) -> bool:
    return any(start <= ord(char) <= end for char in text)


def _resolve_turn_language(content: str, detected_language: str, fallback_language: str = "en") -> str:
    """Resolve turn language from transcript content first, then detector metadata."""
    detected_base = _base_language_code(detected_language)
    text = (content or "").strip()
    if not text:
        return detected_base or _base_language_code(fallback_language) or "en"

    if _contains_codepoint(text, 0x0C80, 0x0CFF):
        return "kn"
    if _contains_codepoint(text, 0x0B80, 0x0BFF):
        return "ta"
    if _contains_codepoint(text, 0x0C00, 0x0C7F):
        return "te"
    if _contains_codepoint(text, 0x0D00, 0x0D7F):
        return "ml"
    if _contains_codepoint(text, 0x0A00, 0x0A7F):
        return "pa"
    if _contains_codepoint(text, 0x0B00, 0x0B7F):
        return "or"
    if _contains_codepoint(text, 0x0600, 0x06FF):
        return "ur" if detected_base == "ur" else detected_base or "ur"
    if _contains_codepoint(text, 0x0980, 0x09FF):
        if detected_base in _BENGALI_ASSAMESE_LANGUAGE_CODES:
            return detected_base
        return "bn"
    if _contains_codepoint(text, 0x0900, 0x097F):
        if detected_base in _DEVANAGARI_LANGUAGE_CODES:
            return detected_base
        return "hi"

    latin_letters = [char for char in text if char.isalpha() and char.isascii()]
    non_latin_letters = [char for char in text if char.isalpha() and not char.isascii()]
    if latin_letters and not non_latin_letters:
        tokens = {
            token.strip(".,!?;:'\"()[]{}").lower()
            for token in text.split()
            if token.strip(".,!?;:'\"()[]{}")
        }
        english_hits = len(tokens & _COMMON_ENGLISH_TOKENS)
        if detected_base == "en" or english_hits > 0:
            return "en"
        if detected_base and detected_base not in _INDIAN_LANGUAGE_CODES:
            return detected_base
        if len(tokens) >= 3:
            return "en"

    return detected_base or _base_language_code(fallback_language) or "en"


AGENT_GREETING = os.getenv("DEEPGRAM_AGENT_GREETING")
AGENT_PROMPT = os.getenv(
    "DEEPGRAM_AGENT_PROMPT",
    (
        "You are a concise, helpful multilingual voice assistant. Answer the "
        "user's actual question directly, including general knowledge, casual, "
        "and everyday questions. Always reply only in the same language the "
        "user used. Do not add English translations, bilingual restatements, "
        "or extra language variants unless the user explicitly asks for "
        "translation. Keep replies natural for spoken conversation."
    ),
)
LANGUAGE_POLICY_PROMPT = (
    "Language policy: reply in the exact same language as the user's latest "
    "message. If the user speaks English, reply in English. Never default to "
    "Hindi or any other language just because the speaker has an Indian accent. "
    "Use the transcript content itself as the primary language signal, and use "
    "detector metadata only as a fallback when the transcript is ambiguous. "
    "Answer the user's request directly instead of just acknowledging it. "
    "Support general questions, factual questions, casual conversation, and "
    "task-oriented requests in the same detected language. Do not translate, "
    "do not mix languages, and do not add bilingual repeats unless the user "
    "explicitly asks for translation."
)
EFFECTIVE_AGENT_PROMPT = f"{LANGUAGE_POLICY_PROMPT}\n\n{AGENT_PROMPT}".strip()

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


@dataclass
class ProxySessionState:
    ready_to_stream: asyncio.Event
    agent_speaking: asyncio.Event
    current_lang: str = ""
    turn_started_at: float | None = None
    first_audio_at: float | None = None
    last_agent_started_at: float | None = None

    def mark_turn_started(self) -> None:
        self.turn_started_at = time.perf_counter()
        self.first_audio_at = None
        self.last_agent_started_at = None

    def mark_agent_started(self) -> None:
        self.last_agent_started_at = time.perf_counter()

    def mark_first_audio(self) -> float | None:
        if self.turn_started_at is None or self.first_audio_at is not None:
            return None
        self.first_audio_at = time.perf_counter()
        return self.first_audio_at - self.turn_started_at


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
        # Use aiohttp for the Deepgram WebSocket — it properly serializes
        # concurrent sends via asyncio, eliminating ping/pong write conflicts.
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                DEEPGRAM_AGENT_URL,
                headers=headers,
                heartbeat=None,          # disable aiohttp's own keepalive
                receive_timeout=None,    # no read timeout
                max_msg_size=0,          # no message size limit for audio
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


async def run_proxy(
    client_ws: WebSocket,
    deepgram_ws: aiohttp.ClientWebSocketResponse,
) -> None:
    state = ProxySessionState(
        ready_to_stream=asyncio.Event(),
        agent_speaking=asyncio.Event(),
    )

    # ----------------------------------------------------------------
    # Single-writer queue → all outgoing Deepgram sends go through here.
    # Eliminates concurrent-write frame corruption AND removes the lock
    # that was blocking audio sends behind control messages.
    # ----------------------------------------------------------------
    write_queue: asyncio.Queue[bytes | str | None] = asyncio.Queue()

    # 100ms of 16-bit silence at 16000 Hz — injected to prevent
    # CLIENT_MESSAGE_TIMEOUT when echo-cancellation suppresses the mic.
    _SILENCE = bytes(3200)

    async def writer() -> None:
        """Dedicated single-writer coroutine — drains write_queue to Deepgram."""
        while True:
            try:
                # Wait up to 100 ms for the next payload
                data = await asyncio.wait_for(write_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # No audio for 100 ms — inject silence if agent is speaking
                # so Deepgram never hits CLIENT_MESSAGE_TIMEOUT.
                if state.agent_speaking.is_set() and state.ready_to_stream.is_set():
                    try:
                        await deepgram_ws.send_bytes(_SILENCE)
                    except Exception:
                        pass
                continue

            if data is None:          # sentinel → shut down writer
                break
            try:
                if isinstance(data, bytes):
                    await deepgram_ws.send_bytes(data)
                else:
                    await deepgram_ws.send_str(data)
            except Exception as e:
                print(f"Deepgram write error: {e}", flush=True)
            finally:
                write_queue.task_done()

    async def send_to_deepgram(data: str | bytes) -> None:
        """Enqueue data for the single writer — non-blocking, never delays callers."""
        await write_queue.put(data)

    async def safe_send_browser_bytes(data: bytes) -> None:
        try:
            await client_ws.send_bytes(data)
        except Exception:
            pass

    async def safe_send_browser_json(payload: dict) -> None:
        try:
            await client_ws.send_json(payload)
        except Exception:
            pass

    async def browser_to_deepgram() -> None:
        while True:
            try:
                message = await client_ws.receive()
            except Exception:
                break

            if "bytes" in message and message["bytes"] is not None:
                await state.ready_to_stream.wait()
                await send_to_deepgram(message["bytes"])
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")
                if msg_type == "Settings":
                    await send_to_deepgram(json.dumps(payload))
                elif msg_type == "KeepAlive":
                    await send_to_deepgram(json.dumps({"type": "KeepAlive"}))
                elif msg_type == "InjectUserMessage":
                    await state.ready_to_stream.wait()
                    await send_to_deepgram(json.dumps(payload))
            elif message.get("type") == "websocket.disconnect":
                break

    async def deepgram_to_browser() -> None:
        async for msg in deepgram_ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if state.agent_speaking.is_set():
                    first_audio_latency = state.mark_first_audio()
                    if first_audio_latency is not None:
                        await safe_send_browser_json(
                            {
                                "type": "ProxyMetrics",
                                "first_audio_latency_ms": round(first_audio_latency * 1000),
                            }
                        )
                await safe_send_browser_bytes(msg.data)
                continue

            if msg.type == aiohttp.WSMsgType.ERROR:
                print(f"Deepgram WS error: {deepgram_ws.exception()}", flush=True)
                break

            if msg.type == aiohttp.WSMsgType.CLOSED:
                print("Deepgram WS closed.", flush=True)
                break

            if msg.type == aiohttp.WSMsgType.PING:
                continue  # aiohttp handles pong automatically

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            payload = json.loads(msg.data)
            msg_type = payload.get("type")
            print(f"Deepgram sent: {msg_type}", flush=True)

            if msg_type == "Error":
                print(f"Deepgram Error: {payload}", flush=True)

            await safe_send_browser_json(payload)

            if msg_type == "Welcome":
                settings = default_settings()
                print(f"Agent speak settings: {settings['agent']['speak']}", flush=True)
                await send_to_deepgram(json.dumps(settings))

            elif msg_type == "SettingsApplied":
                state.ready_to_stream.set()
                await safe_send_browser_json(
                    {
                        "type": "ProxySessionInfo",
                        "tts_provider": TTS_PROVIDER,
                        "listen_model": LISTEN_MODEL,
                        "think_model": THINK_MODEL,
                        "speak_model": (
                            ELEVEN_LABS_MODEL_ID
                            if TTS_PROVIDER == "eleven_labs"
                            else SPEAK_MODEL
                        ),
                        "dynamic_voice_routing": ENABLE_ELEVEN_LABS_LANGUAGE_ROUTING,
                        "input_sample_rate": 16000,
                        "output_sample_rate": 24000,
                    }
                )

            elif msg_type == "UserStartedSpeaking":
                state.mark_turn_started()

            elif msg_type == "AgentStartedSpeaking":
                state.agent_speaking.set()
                state.mark_agent_started()

            elif msg_type == "AgentAudioDone":
                state.agent_speaking.clear()

            elif msg_type == "ConversationText" and TTS_PROVIDER == "eleven_labs":
                if (
                    not ENABLE_ELEVEN_LABS_LANGUAGE_ROUTING
                    or payload.get("role") != "user"
                    or state.agent_speaking.is_set()
                ):
                    continue

                detected_lang = payload.get("language") or ""
                content = payload.get("content") or ""
                base_lang = _resolve_turn_language(
                    content,
                    detected_lang,
                    state.current_lang or AGENT_LANGUAGE or "en",
                )
                if not base_lang or base_lang == state.current_lang:
                    continue

                el_lang = _resolved_elevenlabs_tts_language(base_lang)
                voice_id = _elevenlabs_voice_id(base_lang)
                el_url = (
                    f"wss://api.elevenlabs.io/v1/text-to-speech"
                    f"/{voice_id}/multi-stream-input"
                    f"?optimize_streaming_latency={ELEVEN_LABS_OPTIMIZE_STREAMING_LATENCY}"
                )
                eleven_labs_key = os.getenv("ELEVEN_LABS_API_KEY", "")
                update_speak = {
                    "type": "UpdateSpeak",
                    "speak": {
                        "provider": {
                            "type": "eleven_labs",
                            "model_id": ELEVEN_LABS_MODEL_ID,
                            "language": el_lang,
                        },
                        "endpoint": {
                            "url": el_url,
                            "headers": {"xi-api-key": eleven_labs_key},
                        },
                    },
                }
                print(
                    f"Switching voice -> lang={detected_lang} base={base_lang} "
                    f"el_lang={el_lang} voice={voice_id}",
                    flush=True,
                )
                await send_to_deepgram(json.dumps(update_speak))
                state.current_lang = base_lang

        print("Deepgram listener exited.", flush=True)

    writer_task = asyncio.create_task(writer())
    tasks = [
        asyncio.create_task(browser_to_deepgram()),
        asyncio.create_task(deepgram_to_browser()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        try:
            task.result()
        except Exception as e:
            print(f"Proxy task ended with: {e}", flush=True)

    # Gracefully stop the writer
    await write_queue.put(None)
    await writer_task






def default_settings() -> dict[str, Any]:
    eleven_labs_key = os.getenv("ELEVEN_LABS_API_KEY")
    eleven_labs_voice_id = (
        os.getenv("ELEVEN_LABS_VOICE_ID", "").strip()
        or _elevenlabs_voice_id(AGENT_LANGUAGE or "en")
    )

    if TTS_PROVIDER == "eleven_labs":
        if not eleven_labs_key:
            raise RuntimeError("TTS_PROVIDER=eleven_labs requires ELEVEN_LABS_API_KEY.")
        el_lang = _resolved_elevenlabs_tts_language(AGENT_LANGUAGE or "en")
        voice_id = quote(eleven_labs_voice_id, safe="")
        el_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            "/multi-stream-input"
            f"?optimize_streaming_latency={ELEVEN_LABS_OPTIMIZE_STREAMING_LATENCY}"
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
            "prompt": EFFECTIVE_AGENT_PROMPT,
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

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
