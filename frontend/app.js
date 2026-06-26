const SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

const toggle = document.querySelector("#toggle");
const statusEl = document.querySelector("#status");
const transcript = document.querySelector("#transcript");
const meterBar = document.querySelector("#meterBar");
const metaEl = document.querySelector("#meta");


let ws;
let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let playbackCursor = 0;
let keepAliveTimer;
let disconnectReason = "Press Start to open a new voice session";
let sessionId = 0;

toggle.addEventListener("click", async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    stop();
    return;
  }
  await start();
});


async function start() {
  const currentSessionId = ++sessionId;
  const currentWs = new WebSocket(wsUrl("/ws/agent"));
  setStatus("Connecting");
  setMeta("Opening microphone and voice session");
  toggle.disabled = true;
  disconnectReason = "Press Start to open a new voice session";

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

  const nextAudioContext = new Context({ sampleRate: SAMPLE_RATE });
  await nextAudioContext.audioWorklet.addModule("/assets/pcm-worklet.js");
  if (currentSessionId !== sessionId) {
    await nextAudioContext.close();
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
  if (currentSessionId !== sessionId) {
    nextMediaStream.getTracks().forEach((track) => track.stop());
    await nextAudioContext.close();
    return;
  }
  if (ws !== currentWs || currentWs.readyState !== WebSocket.OPEN) {
    nextMediaStream.getTracks().forEach((track) => track.stop());
    await nextAudioContext.close();
    return;
  }

  const pipeline = startMicrophone(nextAudioContext, nextMediaStream, currentWs);
  if (currentSessionId !== sessionId || ws !== currentWs) {
    pipeline.processorNode.disconnect();
    pipeline.sourceNode.disconnect();
    nextMediaStream.getTracks().forEach((track) => track.stop());
    await nextAudioContext.close();
    return;
  }

  audioContext = nextAudioContext;
  mediaStream = nextMediaStream;
  sourceNode = pipeline.sourceNode;
  processorNode = pipeline.processorNode;
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  startKeepAlive();
  setStatus("Session starting");
  toggle.textContent = "Stop";
  toggle.dataset.active = "true";
  toggle.disabled = false;
}

function startMicrophone(nextAudioContext, nextMediaStream, currentWs) {
  const nextSourceNode = nextAudioContext.createMediaStreamSource(nextMediaStream);
  const nextProcessorNode = new AudioWorkletNode(nextAudioContext, "pcm-worklet");
  nextProcessorNode.port.onmessage = (event) => {
    const { pcm, level } = event.data;
    meterBar.style.width = `${Math.min(100, Math.round(level * 140))}%`;
    if (ws === currentWs && currentWs.readyState === WebSocket.OPEN) {
      currentWs.send(pcm);
    }
  };

  nextSourceNode.connect(nextProcessorNode);
  nextProcessorNode.connect(nextAudioContext.destination);

  return {
    sourceNode: nextSourceNode,
    processorNode: nextProcessorNode,
  };
}

function handleEvent(event) {
  if (event.type === "ConversationText") {
    addMessage(event.role, event.content);
  } else if (event.type === "SettingsApplied") {
    setStatus("Voice agent ready");
    setMeta("Live session on Deepgram Voice Agent");
  } else if (event.type === "UserStartedSpeaking") {
    setStatus("Listening");
  } else if (event.type === "AgentThinking") {
    setStatus("Thinking");
    if (event.content) setMeta(event.content);
  } else if (event.type === "AgentStartedSpeaking") {
    setStatus("Speaking");
    setMeta(formatLatency(event));
  } else if (event.type === "AgentAudioDone") {
    setStatus("Listening");
  } else if (event.type === "Welcome") {
    setStatus("Configuring session");
  } else if (event.type === "History") {
    renderHistory(event);
  } else if (event.type === "Error" || event.type === "Warning") {
    disconnectReason = event.description || event.message || event.type;
    setStatus(disconnectReason);
    setMeta(event.code ? `Code: ${event.code}` : "");
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
  if (!audioContext) return;

  const pcm = new Int16Array(arrayBuffer);
  const audioBuffer = audioContext.createBuffer(
    1,
    pcm.length,
    OUTPUT_SAMPLE_RATE,
  );
  const channel = audioBuffer.getChannelData(0);
  for (let index = 0; index < pcm.length; index += 1) {
    channel[index] = pcm[index] / 32768;
  }

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  
  // If playback has fallen behind, add a small 50ms jitter buffer 
  // to prevent continuous micro-stuttering from network latency
  if (playbackCursor < now) {
    playbackCursor = now + 0.05;
  }
  
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
  mediaStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();

  processorNode = undefined;
  sourceNode = undefined;
  mediaStream = undefined;
  audioContext = undefined;
  playbackCursor = 0;

  meterBar.style.width = "0%";
  toggle.textContent = "Start";
  toggle.dataset.active = "false";
  toggle.disabled = false;
  setStatus("Disconnected");
  setMeta(disconnectReason);
}

function setStatus(status) {
  statusEl.textContent = status;
}

function setMeta(text) {
  metaEl.textContent = text;
}

function startKeepAlive() {
  window.clearInterval(keepAliveTimer);
  keepAliveTimer = window.setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "KeepAlive" }));
    }
  }, 15000);
}

function formatLatency(event) {
  const total = Number(event.total_latency ?? 0).toFixed(2);
  const tts = Number(event.tts_latency ?? 0).toFixed(2);
  const think = Number(event.ttt_latency ?? 0).toFixed(2);
  return `Total ${total}s, think ${think}s, speak ${tts}s`;
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
