package com.wenet.asr;

final class WavData {
    final int sampleRate;
    final short[] samples;

    WavData(int sampleRate, short[] samples) {
        this.sampleRate = sampleRate;
        this.samples = samples;
    }
}
