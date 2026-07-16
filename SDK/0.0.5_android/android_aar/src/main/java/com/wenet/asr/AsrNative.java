package com.wenet.asr;

final class AsrNative {
    static {
        System.loadLibrary("onnxruntime");
        System.loadLibrary("asr_sdk");
        System.loadLibrary("asr_jni");
    }

    private AsrNative() {}

    static native long createEngine(String modelDir);
    static native void destroyEngine(long engine);
    static native long createStream(long engine);
    static native void destroyStream(long stream);
    static native void acceptPcm16(
            long stream, short[] samples, int offset, int length, int sampleRate);
    static native boolean decodeReady(long stream);
    static native void decode(long stream);
    static native void setInputFinished(long stream);
    static native void resetStream(long stream);
    static native String getResultJson(long stream);
    static native String getFinalResultJson(long stream);
    static native String version();
    static native int abiVersion();
    static native String buildInfoJson();
}
