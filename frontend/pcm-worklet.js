class PcmWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options?.processorOptions ?? {};
    const packetMs = Number(processorOptions.packetMs ?? 40);
    const sampleRate = Number(processorOptions.sampleRate ?? 16000);
    this.packetFrames = Math.max(128, Math.round((sampleRate * packetMs) / 1000));
    this.packetBuffer = new Int16Array(this.packetFrames);
    this.packetCursor = 0;
    this.levelAccumulator = 0;
    this.levelFrames = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) {
      return true;
    }

    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      this.packetBuffer[this.packetCursor] =
        sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      this.packetCursor += 1;
      this.levelAccumulator += sample * sample;
      this.levelFrames += 1;

      if (this.packetCursor === this.packetFrames) {
        const level = Math.sqrt(this.levelAccumulator / this.levelFrames);
        const pcm = this.packetBuffer.buffer.slice(0);
        this.port.postMessage({ pcm, level }, [pcm]);
        this.packetBuffer = new Int16Array(this.packetFrames);
        this.packetCursor = 0;
        this.levelAccumulator = 0;
        this.levelFrames = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);
