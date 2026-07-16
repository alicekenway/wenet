package com.example.asrdemo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.wenet.asr.AsrRecognizer;
import com.wenet.asr.AsrResult;
import com.wenet.asr.AsrSdk;

import java.io.File;
import java.io.InputStream;

public class MainActivity extends Activity {
    private static final String TAG = "ASR_TEST";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView textView = new TextView(this);
        textView.setText("Running ASR test...");
        setContentView(textView);

        new Thread(new Runnable() {
            @Override
            public void run() {
                String result;
                try {
                    Log.i(TAG, "sdk_version=" + AsrSdk.version());
                    Log.i(TAG, "build_info=" + AsrSdk.buildInfoJson());

                    try (InputStream modelZip = getAssets().open("asr-model.zip")) {
                        AsrSdk.installModelZip(MainActivity.this, modelZip);
                    }

                    File wavFile = DemoAssets.copyAsset(
                            MainActivity.this, "test.wav", "test.wav");
                    try (AsrRecognizer recognizer =
                                 AsrSdk.createRecognizer(MainActivity.this)) {
                        AsrResult asrResult = recognizer.transcribeWav(wavFile);
                        Log.i(TAG, "final_result=" + asrResult.getJson());
                        Log.i(TAG, "final_text=" + asrResult.getText());
                        result = asrResult.getText() + "\n\n" + asrResult.getJson();
                    }
                } catch (Throwable t) {
                    result = "ERROR: " + t;
                    Log.e(TAG, result, t);
                }
                final String display = result;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        textView.setText(display);
                    }
                });
            }
        }).start();
    }
}
