package com.example.asrdemo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

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
                    result = AsrSdkUsageExample.runPackagedWav(MainActivity.this);
                    Log.i(TAG, "final_result=" + result);
                    Log.i(TAG, "final_text=" + AsrSdkUsageExample.extractText(result));
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
