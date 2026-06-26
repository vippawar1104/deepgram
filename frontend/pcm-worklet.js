class PcmWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    const pcm = new Int16Array(input.length);
    let sum = 0;
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      sum += sample * sample;
    }

    const level = Math.sqrt(sum / input.length);
    this.port.postMessage({ pcm: pcm.buffer, level }, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);
