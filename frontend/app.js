const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const PCM_PACKET_MS = 40;
const JITTER_BUFFER_MS = 60;

const toggle = document.querySelector("#toggle");
const composer = document.querySelector("#composer");
const textPrompt = document.querySelector("#textPrompt");
const sendTextBtn = document.querySelector("#sendText");
const statusEl = document.querySelector("#status");
const transcript = document.querySelector("#transcript");
const meterBar = document.querySelector("#meterBar");
const metaEl = document.querySelector("#meta");
const firstAudioMetric = document.querySelector("#firstAudioMetric");
const thinkMetric = document.querySelector("#thinkMetric");
const speakMetric = document.querySelector("#speakMetric");
const packetMetric = document.querySelector("#packetMetric");
const queueMetric = document.querySelector("#queueMetric");
const ttsMetric = document.querySelector("#ttsMetric");
const sessionMetric = document.querySelector("#sessionMetric");
const captureStage = document.querySelector("#captureStage");
const listenStage = document.querySelector("#listenStage");
const thinkStage = document.querySelector("#thinkStage");
const speakStage = document.querySelector("#speakStage");

let ws;
let inputContext;
let playbackContext;
let mediaStream;
let sourceNode;
let processorNode;
let captureSink;
let playbackCursor = 0;
let keepAliveTimer;
let disconnectReason = "Press Start to open a new voice session";
let sessionId = 0;
let turnState = resetTurnState();
let sessionInfo = null;
let pendingTextTurn = false;
let pendingUserEcho = "";

packetMetric.textContent = `${PCM_PACKET_MS} ms`;
setComposerEnabled(false);

toggle.addEventListener("click", async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    stop();
    return;
  }
  await start();
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendTypedMessage();
});

textPrompt.addEventListener("input", () => {
  syncComposerState();
});

textPrompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void sendTypedMessage();
  }
});

async function start() {
  const currentSessionId = ++sessionId;
  const currentWs = new WebSocket(wsUrl("/ws/agent"));
  setStatus("Connecting");
  setMeta("Opening microphone and voice session");
  setSessionMetric("Handshake in progress");
  setStageState(captureStage, "warn");
  toggle.disabled = true;
  setComposerEnabled(false);
  disconnectReason = "Press Start to open a new voice session";
  turnState = resetTurnState();
  pendingTextTurn = false;
  resetMetrics();

  try {
    ws = currentWs;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      startAudioPipeline(currentSessionId, currentWs).catch((error) => {
        if (currentSessionId !== sessionId) return;
        disconnectReason = error?.message || "Failed to start microphone";
        setStatus("Start failed");
        setMeta(disconnectReason);
        currentWs.close();
      });
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        handleEvent(JSON.parse(event.data));
        return;
      }

      const audioBytes =
        event.data instanceof Blob
          ? await event.data.arrayBuffer()
          : event.data;
      playPcm(audioBytes);
    };

    ws.onclose = (event) => {
      if (currentSessionId !== sessionId) return;
      if (event.reason) {
        disconnectReason = event.reason;
      }
      stop(false);
    };

    ws.onerror = () => {
      disconnectReason = "WebSocket connection failed";
      setStatus("Connection error");
      setMeta(disconnectReason);
    };
  } catch (error) {
    disconnectReason = error?.message || "Failed to start voice session";
    stop(false);
    setStatus("Start failed");
    setMeta(disconnectReason);
  }
}

async function startAudioPipeline(currentSessionId, currentWs) {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) {
    throw new Error("This browser does not support Web Audio.");
  }

  const nextInputContext = new Context({ sampleRate: INPUT_SAMPLE_RATE });
  await nextInputContext.audioWorklet.addModule("/assets/pcm-worklet.js");
  const nextPlaybackContext = new Context();
  if (currentSessionId !== sessionId) {
    await closeContext(nextInputContext);
    await closeContext(nextPlaybackContext);
    return;
  }

  const nextMediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  if (currentSessionId !== sessionId || ws !== currentWs || currentWs.readyState !== WebSocket.OPEN) {
    nextMediaStream.getTracks().forEach((track) => track.stop());
    await closeContext(nextInputContext);
    await closeContext(nextPlaybackContext);
    return;
  }

  const pipeline = startMicrophone(nextInputContext, nextMediaStream, currentWs);
  if (currentSessionId !== sessionId || ws !== currentWs) {
    pipeline.processorNode.disconnect();
    pipeline.sourceNode.disconnect();
    pipeline.captureSink.disconnect();
    nextMediaStream.getTracks().forEach((track) => track.stop());
    await closeContext(nextInputContext);
    await closeContext(nextPlaybackContext);
    return;
  }

  inputContext = nextInputContext;
  playbackContext = nextPlaybackContext;
  mediaStream = nextMediaStream;
  sourceNode = pipeline.sourceNode;
  processorNode = pipeline.processorNode;
  captureSink = pipeline.captureSink;
  if (inputContext.state === "suspended") {
    await inputContext.resume();
  }
  if (playbackContext.state === "suspended") {
    await playbackContext.resume();
  }

  startKeepAlive();
  setStatus("Session starting");
  setMeta("Negotiating Deepgram voice agent settings");
  setSessionMetric("Audio capture live");
  toggle.textContent = "Stop";
  toggle.dataset.active = "true";
  toggle.disabled = false;
  setStageState(captureStage, "active");
  setComposerEnabled(true);
}

function startMicrophone(nextInputContext, nextMediaStream, currentWs) {
  const nextSourceNode = nextInputContext.createMediaStreamSource(nextMediaStream);
  const nextProcessorNode = new AudioWorkletNode(nextInputContext, "pcm-worklet", {
    processorOptions: {
      packetMs: PCM_PACKET_MS,
      sampleRate: INPUT_SAMPLE_RATE,
    },
  });
  const nextCaptureSink = nextInputContext.createGain();
  nextCaptureSink.gain.value = 0;

  nextProcessorNode.port.onmessage = (event) => {
    const { pcm, level } = event.data;
    meterBar.style.width = `${Math.min(100, Math.round(level * 160))}%`;
    if (ws === currentWs && currentWs.readyState === WebSocket.OPEN) {
      currentWs.send(pcm);
    }
  };

  nextSourceNode.connect(nextProcessorNode);
  nextProcessorNode.connect(nextCaptureSink);
  nextCaptureSink.connect(nextInputContext.destination);

  return {
    sourceNode: nextSourceNode,
    processorNode: nextProcessorNode,
    captureSink: nextCaptureSink,
  };
}

function handleEvent(event) {
  if (event.type === "ConversationText") {
    if (event.role === "user") {
      if (pendingUserEcho && event.content === pendingUserEcho) {
        pendingUserEcho = "";
      } else {
        addMessage(event.role, event.content);
      }
      pendingTextTurn = false;
      setStageState(listenStage, "done");
      setStageState(thinkStage, "active");
      turnState.transcriptAt = performance.now();
      setSessionMetric("User turn transcribed");
      syncComposerState();
      return;
    }
    addMessage(event.role, event.content);
    return;
  }

  if (event.type === "SettingsApplied") {
    setStatus("Voice agent ready");
    setMeta("Live session on Deepgram Voice Agent");
    setSessionMetric("Waiting for speech");
    setStageState(captureStage, "active");
    setStageState(listenStage, "");
    setComposerEnabled(true);
    return;
  }

  if (event.type === "ProxySessionInfo") {
    sessionInfo = event;
    ttsMetric.textContent = formatTtsMetric(event);
    return;
  }

  if (event.type === "UserStartedSpeaking") {
    turnState = resetTurnState();
    turnState.userSpeechStartAt = performance.now();
    pendingTextTurn = false;
    setStatus("Listening");
    setMeta("Streaming microphone audio to Deepgram");
    setSessionMetric("User turn in progress");
    setStageState(captureStage, "active");
    setStageState(listenStage, "active");
    setStageState(thinkStage, "");
    setStageState(speakStage, "");
    return;
  }

  if (event.type === "AgentThinking") {
    setStatus("Thinking");
    setMeta(event.content || "Generating response");
    setSessionMetric("Language model running");
    if (turnState.transcriptAt) {
      turnState.thinkStartAt = performance.now();
    }
    setStageState(thinkStage, "active");
    return;
  }

  if (event.type === "AgentStartedSpeaking") {
    turnState.agentStartedAt = performance.now();
    setStatus("Speaking");
    setMeta(formatLatency(event));
    setSessionMetric("Receiving synthesized audio");
    setStageState(thinkStage, "done");
    setStageState(speakStage, "active");
    if (Number.isFinite(Number(event.ttt_latency))) {
      thinkMetric.textContent = formatSeconds(Number(event.ttt_latency));
    }
    if (Number.isFinite(Number(event.tts_latency))) {
      speakMetric.textContent = formatSeconds(Number(event.tts_latency));
    }
    return;
  }

  if (event.type === "ProxyMetrics") {
    if (typeof event.first_audio_latency_ms === "number") {
      firstAudioMetric.textContent = `${event.first_audio_latency_ms} ms`;
    }
    return;
  }

  if (event.type === "AgentAudioDone") {
    setStatus("Listening");
    setMeta("Ready for the next turn");
    setSessionMetric("Turn complete");
    setStageState(listenStage, "");
    setStageState(thinkStage, "");
    setStageState(speakStage, "done");
    syncComposerState();
    return;
  }

  if (event.type === "Welcome") {
    setStatus("Configuring session");
    setMeta("Applying backend and provider settings");
    setSessionMetric("Connected to Deepgram");
    return;
  }

  if (event.type === "History") {
    renderHistory(event);
    return;
  }

  if (event.type === "Error" || event.type === "Warning") {
    pendingTextTurn = false;
    disconnectReason = event.description || event.message || event.type;
    setStatus(disconnectReason);
    setMeta(event.code ? `Code: ${event.code}` : "");
    setSessionMetric("Session interrupted");
    syncComposerState();
  }
}

function addMessage(role, content) {
  const item = document.createElement("div");
  item.className = `message ${role === "user" ? "user" : "agent"}`;
  item.innerHTML = `<span class="role">${escapeHtml(role)}</span>${escapeHtml(content)}`;
  transcript.append(item);
  transcript.scrollTop = transcript.scrollHeight;
}

function playPcm(arrayBuffer) {
  if (!playbackContext) return;

  const pcm = new Int16Array(arrayBuffer);
  const audioBuffer = playbackContext.createBuffer(1, pcm.length, OUTPUT_SAMPLE_RATE);
  const channel = audioBuffer.getChannelData(0);
  for (let index = 0; index < pcm.length; index += 1) {
    channel[index] = pcm[index] / 32768;
  }

  const source = playbackContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  if (playbackCursor < now) {
    playbackCursor = now + JITTER_BUFFER_MS / 1000;
  }

  const queueMs = Math.max(0, Math.round((playbackCursor - now) * 1000));
  queueMetric.textContent = `${queueMs} ms`;
  source.start(playbackCursor);
  playbackCursor += audioBuffer.duration;
}

function stop(closeSocket = true) {
  sessionId += 1;
  if (closeSocket && ws?.readyState === WebSocket.OPEN) {
    ws.close();
  }
  ws = undefined;
  window.clearInterval(keepAliveTimer);
  keepAliveTimer = undefined;

  processorNode?.disconnect();
  sourceNode?.disconnect();
  captureSink?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  void closeContext(inputContext);
  void closeContext(playbackContext);

  processorNode = undefined;
  sourceNode = undefined;
  captureSink = undefined;
  mediaStream = undefined;
  inputContext = undefined;
  playbackContext = undefined;
  playbackCursor = 0;
  turnState = resetTurnState();
  sessionInfo = null;
  pendingTextTurn = false;
  pendingUserEcho = "";

  meterBar.style.width = "0%";
  queueMetric.textContent = "0 ms";
  toggle.textContent = "Start";
  toggle.dataset.active = "false";
  toggle.disabled = false;
  setComposerEnabled(false);
  setStatus("Disconnected");
  setMeta(disconnectReason);
  setSessionMetric("Idle");
  resetStageState();
}

function setStatus(status) {
  statusEl.textContent = status;
}

function setMeta(text) {
  metaEl.textContent = text;
}

function setSessionMetric(text) {
  sessionMetric.textContent = text;
}

function resetMetrics() {
  firstAudioMetric.textContent = "--";
  thinkMetric.textContent = "--";
  speakMetric.textContent = "--";
  queueMetric.textContent = "0 ms";
  ttsMetric.textContent = "--";
}

async function sendTypedMessage() {
  const content = textPrompt.value.trim();
  if (!content || !isSessionReady() || pendingTextTurn) {
    return;
  }

  pendingTextTurn = true;
  turnState = resetTurnState();
  turnState.userSpeechStartAt = performance.now();
  setStatus("Sending");
  setMeta("Submitting typed message to Deepgram");
  setSessionMetric("Typed turn in progress");
  setStageState(captureStage, "done");
  setStageState(listenStage, "done");
  setStageState(thinkStage, "active");
  setStageState(speakStage, "");

  try {
    ws.send(
      JSON.stringify({
        type: "InjectUserMessage",
        content: content,
      }),
    );
    pendingUserEcho = content;
    addMessage("user", content);
    textPrompt.value = "";
    syncComposerState();
  } catch (error) {
    pendingTextTurn = false;
    pendingUserEcho = "";
    setStatus("Send failed");
    setMeta(error?.message || "Failed to send typed message");
    syncComposerState();
  }
}

function startKeepAlive() {
  window.clearInterval(keepAliveTimer);
  keepAliveTimer = window.setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "KeepAlive" }));
    }
  }, 5000);
}

function isSessionReady() {
  return Boolean(ws && ws.readyState === WebSocket.OPEN);
}

function formatLatency(event) {
  const total = Number(event.total_latency ?? 0);
  const tts = Number(event.tts_latency ?? 0);
  const think = Number(event.ttt_latency ?? 0);
  return `Total ${total.toFixed(2)} s, think ${think.toFixed(2)} s, speak ${tts.toFixed(2)} s`;
}

function formatTtsMetric(event) {
  const provider = event.tts_provider === "eleven_labs" ? "ElevenLabs" : "Deepgram";
  const routing = event.dynamic_voice_routing ? "dynamic" : "fixed";
  return `${provider} ${routing}`;
}

function formatSeconds(value) {
  return `${value.toFixed(2)} s`;
}

function renderHistory(event) {
  if (event.role && event.content) {
    addMessage(event.role, event.content);
  }
}

function wsUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${path}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resetTurnState() {
  return {
    userSpeechStartAt: null,
    transcriptAt: null,
    thinkStartAt: null,
    agentStartedAt: null,
  };
}

function resetStageState() {
  [captureStage, listenStage, thinkStage, speakStage].forEach((node) => {
    node.dataset.state = "";
  });
}

function setStageState(node, state) {
  node.dataset.state = state;
}

async function closeContext(context) {
  if (!context || context.state === "closed") return;
  await context.close();
}

function setComposerEnabled(enabled) {
  textPrompt.disabled = !enabled;
  syncComposerState();
}

function syncComposerState() {
  sendTextBtn.disabled = !isSessionReady() || !textPrompt.value.trim() || pendingTextTurn;
}
