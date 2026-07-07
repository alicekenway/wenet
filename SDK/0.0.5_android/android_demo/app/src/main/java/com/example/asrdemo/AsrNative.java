package com.example.asrdemo;

public final class AsrNative {
    static {
        System.loadLibrary("onnxruntime");
        System.loadLibrary("asr_sdk");
        System.loadLibrary("asr_jni");
    }

    private AsrNative() {}

    public static native String runWavTest(String modelDir, String wavPath);
}
