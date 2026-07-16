package com.wenet.asr;

public class AsrException extends RuntimeException {
    public AsrException(String message) {
        super(message);
    }

    public AsrException(String message, Throwable cause) {
        super(message, cause);
    }
}
