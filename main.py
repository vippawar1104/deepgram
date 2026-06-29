import argparse
import contextlib
import json
import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterable

import requests
from deepgram import DeepgramClient
from deepgram.agent.v1.types import (
    AgentV1Settings,
    AgentV1SettingsAgent,
    AgentV1SettingsAgentListen,
    AgentV1SettingsAgentListenProvider_V1,
    AgentV1SettingsAudio,
    AgentV1SettingsAudioInput,
    AgentV1SettingsAudioOutput,
)
from deepgram.core.events import EventType
from app import app



DEFAULT_INPUT_URL = "https://dpgr.am/spacewalk.wav"
DEFAULT_PROMPT = (
    "You are a friendly, concise voice assistant. Answer naturally for spoken "
    "conversation and keep responses brief unless the user asks for detail."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Deepgram Agent speech-to-speech demo from WAV input."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Local mono 16-bit PCM WAV file to stream to the voice agent.",
    )
    input_group.add_argument(
        "--input-url",
        default=DEFAULT_INPUT_URL,
        help=f"WAV URL to stream to the voice agent. Defaults to {DEFAULT_INPUT_URL}",
    )
    parser.add_argument("--listen-model", default="nova-3")
    parser.add_argument("--think-model", default="gpt-4o-mini")
    parser.add_argument("--speak-model", default="aura-2-thalia-en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--chatlog", type=Path, default=Path("chatlog.txt"))
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--greeting",
        default=None,
        help="Optional greeting for the agent to speak after settings are applied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPGRAM_API_KEY before running this script.")

    input_path = resolve_input_wav(args)
    input_info = inspect_wav(input_path)
    output_sample_rate = 24_000

    client = DeepgramClient(api_key=api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.chatlog.parent.mkdir(parents=True, exist_ok=True)

    with client.agent.v1.connect() as connection:
        print("Connected to Deepgram Agent API.")

        state = {
            "audio": bytearray(),
            "audio_done_count": 0,
            "file_counter": 0,
            "stop": False,
            "received_audio": False,
        }
        welcome_event = threading.Event()
        settings_applied_event = threading.Event()

        def on_message(message):
            if isinstance(message, bytes):
                state["audio"].extend(message)
                state["received_audio"] = True
                return

            msg_type = getattr(message, "type", None)
            if msg_type == "Welcome":
                welcome_event.set()
            elif msg_type == "SettingsApplied":
                settings_applied_event.set()
            elif msg_type == "ConversationText":
                role = getattr(message, "role", "unknown")
                content = getattr(message, "content", "")
                print(f"{role}: {content}")
                append_jsonl(args.chatlog, model_to_dict(message))
            elif msg_type == "AgentAudioDone":
                if state["audio"]:
                    output_path = args.output_dir / f"output-{state['file_counter']}.wav"
                    write_pcm_wav(output_path, state["audio"], output_sample_rate)
                    print(f"Wrote {output_path}")
                state["audio"] = bytearray()
                state["audio_done_count"] += 1
                state["file_counter"] += 1
            elif msg_type == "Error":
                print(f"Deepgram error: {message}")
                state["stop"] = True
            elif msg_type == "Warning":
                print(f"Deepgram warning: {message}")

        def on_error(error):
            print(f"WebSocket error: {error}")
            state["stop"] = True

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        listener_thread = threading.Thread(
            target=connection.start_listening,
            name="deepgram-agent-listener",
            daemon=True,
        )
        listener_thread.start()
        wait_for_event(welcome_event, timeout=args.timeout, name="Welcome")

        settings = build_agent_settings(
            input_sample_rate=input_info["sample_rate"],
            output_sample_rate=output_sample_rate,
            language=args.language,
            listen_model=args.listen_model,
            think_model=args.think_model,
            speak_model=args.speak_model,
            prompt=args.prompt,
            greeting=args.greeting,
        )
        connection.send_settings(settings)
        wait_for_event(settings_applied_event, timeout=args.timeout, name="SettingsApplied")

        baseline_audio_done_count = state["audio_done_count"]
        print(
            "Streaming "
            f"{input_path} ({input_info['sample_rate']} Hz, "
            f"{input_info['channels']} channel, {input_info['sample_width'] * 8}-bit PCM)."
        )
        for chunk in iter_wav_pcm_chunks(input_path, chunk_ms=args.chunk_ms):
            connection.send_media(chunk)
            time.sleep(args.chunk_ms / 1000)

        wait_for_agent_audio_done(
            state,
            baseline_audio_done_count=baseline_audio_done_count,
            timeout=args.timeout,
        )
        if not state["received_audio"]:
            print("No agent audio was received before timeout.")


def build_agent_settings(
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    language: str,
    listen_model: str,
    think_model: str,
    speak_model: str,
    prompt: str,
    greeting: "Optional[str]",
) -> AgentV1Settings:
    return AgentV1Settings(
        audio=AgentV1SettingsAudio(
            input=AgentV1SettingsAudioInput(
                encoding="linear16",
                sample_rate=input_sample_rate,
            ),
            output=AgentV1SettingsAudioOutput(
                encoding="linear16",
                sample_rate=output_sample_rate,
                container="none",
            ),
        ),
        agent=AgentV1SettingsAgent(
            language=language,
            listen=AgentV1SettingsAgentListen(
                provider=AgentV1SettingsAgentListenProvider_V1(
                    type="deepgram",
                    model=listen_model,
                )
            ),
            think={
                "provider": {
                    "type": "open_ai",
                    "model": think_model,
                },
                "prompt": prompt,
            },
            speak={
                "provider": {
                    "type": "deepgram",
                    "model": speak_model,
                }
            },
            greeting=greeting,
        ),
    )


def resolve_input_wav(args: argparse.Namespace) -> Path:
    if args.input_file:
        return args.input_file

    response = requests.get(args.input_url, timeout=30)
    response.raise_for_status()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with temp:
        temp.write(response.content)
    return Path(temp.name)


def inspect_wav(path: Path) -> dict[str, int]:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        info = {
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "sample_rate": wav_file.getframerate(),
        }

    if info["channels"] != 1:
        raise ValueError("Input WAV must be mono. Convert it before streaming.")
    if info["sample_width"] != 2:
        raise ValueError("Input WAV must be 16-bit PCM.")
    return info


def iter_wav_pcm_chunks(path: Path, *, chunk_ms: int) -> Iterable[bytes]:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        frames_per_chunk = max(1, int(wav_file.getframerate() * chunk_ms / 1000))
        while True:
            chunk = wav_file.readframes(frames_per_chunk)
            if not chunk:
                break
            yield chunk


def write_pcm_wav(path: Path, pcm: bytes | bytearray, sample_rate: int) -> None:
    with contextlib.closing(wave.open(str(path), "wb")) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))


def wait_for_event(event: threading.Event, *, timeout: float, name: str) -> None:
    if not event.wait(timeout):
        raise TimeoutError(f"Timed out waiting for Deepgram {name} message.")


def wait_for_agent_audio_done(
    state: dict,
    *,
    baseline_audio_done_count: int,
    timeout: float,
) -> None:
    start_time = time.monotonic()
    while state["audio_done_count"] <= baseline_audio_done_count and not state["stop"]:
        if time.monotonic() - start_time >= timeout:
            break
        time.sleep(0.25)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as chatlog:
        chatlog.write(json.dumps(payload, ensure_ascii=True) + "\n")


def model_to_dict(message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if hasattr(message, "dict"):
        return message.dict()
    return getattr(message, "__dict__", {"message": str(message)})


if __name__ == "__main__":
    main()
