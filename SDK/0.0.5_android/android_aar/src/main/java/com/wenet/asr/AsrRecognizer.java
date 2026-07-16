package com.wenet.asr;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;

public final class AsrRecognizer implements Closeable {
    private static final int DEFAULT_CHUNK_SAMPLES = 16000;

    private long engine;
    private long stream;

    AsrRecognizer(File modelDir) {
        engine = AsrNative.createEngine(modelDir.getAbsolutePath());
        stream = AsrNative.createStream(engine);
    }

    public AsrResult transcribeWav(File wavFile) throws IOException {
        WavData wav = WavReader.readPcm16(wavFile);
        reset();
        for (int offset = 0; offset < wav.samples.length; offset += DEFAULT_CHUNK_SAMPLES) {
            int length = Math.min(DEFAULT_CHUNK_SAMPLES, wav.samples.length - offset);
            acceptPcm16(wav.samples, offset, length, wav.sampleRate);
            decode();
        }
        return finish();
    }

    public void acceptPcm16(short[] samples, int sampleRate) {
        acceptPcm16(samples, 0, samples.length, sampleRate);
    }

    public void acceptPcm16(short[] samples, int offset, int length, int sampleRate) {
        ensureOpen();
        AsrNative.acceptPcm16(stream, samples, offset, length, sampleRate);
    }

    public AsrResult decode() {
        ensureOpen();
        while (AsrNative.decodeReady(stream)) {
            AsrNative.decode(stream);
        }
        return AsrResult.fromJson(AsrNative.getResultJson(stream));
    }

    public AsrResult finish() {
        ensureOpen();
        AsrNative.setInputFinished(stream);
        while (AsrNative.decodeReady(stream)) {
            AsrNative.decode(stream);
        }
        return AsrResult.fromJson(AsrNative.getFinalResultJson(stream));
    }

    public void reset() {
        ensureOpen();
        AsrNative.resetStream(stream);
    }

    @Override
    public void close() {
        if (stream != 0) {
            AsrNative.destroyStream(stream);
            stream = 0;
        }
        if (engine != 0) {
            AsrNative.destroyEngine(engine);
            engine = 0;
        }
    }

    private void ensureOpen() {
        if (engine == 0 || stream == 0) {
            throw new AsrException("ASR recognizer is closed");
        }
    }
}
