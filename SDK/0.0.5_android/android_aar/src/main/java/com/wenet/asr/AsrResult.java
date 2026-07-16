package com.wenet.asr;

import org.json.JSONException;
import org.json.JSONObject;

public final class AsrResult {
    private final String json;
    private final String text;

    static AsrResult fromJson(String json) {
        if (json == null) {
            return new AsrResult("", "");
        }
        String parsedText = "";
        try {
            JSONObject object = new JSONObject(json);
            parsedText = object.optString("text", "");
        } catch (JSONException ignored) {
            parsedText = "";
        }
        return new AsrResult(json, parsedText);
    }

    private AsrResult(String json, String text) {
        this.json = json;
        this.text = text;
    }

    public String getJson() {
        return json;
    }

    public String getText() {
        return text;
    }

    @Override
    public String toString() {
        return json;
    }
}
