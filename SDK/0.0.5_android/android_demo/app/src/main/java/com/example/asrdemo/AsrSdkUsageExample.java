package com.example.asrdemo;

import android.content.Context;

import java.io.File;
import java.io.IOException;

public final class AsrSdkUsageExample {
    private AsrSdkUsageExample() {}

    public static String runPackagedWav(Context context) throws IOException {
        File modelDir = new File(context.getFilesDir(), "asr_model");
        File wavFile = new File(context.getFilesDir(), "test.wav");
        AssetCopy.copyTree(context, "model", modelDir);
        AssetCopy.copyFile(context, "test.wav", wavFile);
        return runFromFiles(modelDir, wavFile);
    }

    public static String runFromFiles(File modelDir, File wavFile) {
        return AsrNative.runWavTest(
                modelDir.getAbsolutePath(), wavFile.getAbsolutePath());
    }

    public static String extractText(String resultJson) {
        String key = "\"text\":\"";
        int start = resultJson.indexOf(key);
        if (start < 0) {
            return "";
        }
        start += key.length();
        StringBuilder out = new StringBuilder();
        boolean escaping = false;
        for (int i = start; i < resultJson.length(); ++i) {
            char c = resultJson.charAt(i);
            if (escaping) {
                out.append(c);
                escaping = false;
            } else if (c == '\\') {
                escaping = true;
            } else if (c == '"') {
                break;
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }
}
