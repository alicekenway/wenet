package com.wenet.asr;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

final class WavReader {
    private WavReader() {}

    static WavData readPcm16(File file) throws IOException {
        try (InputStream in = new FileInputStream(file)) {
            return readPcm16(in);
        }
    }

    static WavData readPcm16(InputStream in) throws IOException {
        if (!"RIFF".equals(readTag(in))) {
            throw new IOException("missing RIFF header");
        }
        skipFully(in, 4);
        if (!"WAVE".equals(readTag(in))) {
            throw new IOException("missing WAVE header");
        }

        int audioFormat = 0;
        int channels = 0;
        int sampleRate = 0;
        int bitsPerSample = 0;
        byte[] pcmBytes = null;

        while (audioFormat == 0 || pcmBytes == null) {
            String tag = readTagOrNull(in);
            if (tag == null) {
                break;
            }
            int size = readU32(in);
            if ("fmt ".equals(tag)) {
                audioFormat = readU16(in);
                channels = readU16(in);
                sampleRate = readU32(in);
                skipFully(in, 6);
                bitsPerSample = readU16(in);
                if (size > 16) {
                    skipFully(in, size - 16);
                }
            } else if ("data".equals(tag)) {
                pcmBytes = readBytes(in, size);
            } else {
                skipFully(in, size);
            }
            if ((size & 1) == 1) {
                skipFully(in, 1);
            }
        }

        if (audioFormat != 1 || bitsPerSample != 16 || channels <= 0) {
            throw new IOException("only 16-bit PCM WAV is supported");
        }
        if (pcmBytes == null) {
            throw new IOException("missing WAV data chunk");
        }

        int totalSamples = pcmBytes.length / 2;
        int monoSamples = totalSamples / channels;
        short[] samples = new short[monoSamples];
        for (int i = 0; i < monoSamples; ++i) {
            int mixed = 0;
            for (int c = 0; c < channels; ++c) {
                int byteIndex = (i * channels + c) * 2;
                int lo = pcmBytes[byteIndex] & 0xff;
                int hi = pcmBytes[byteIndex + 1] << 8;
                mixed += (short) (lo | hi);
            }
            samples[i] = (short) (mixed / channels);
        }
        return new WavData(sampleRate, samples);
    }

    private static String readTag(InputStream in) throws IOException {
        byte[] bytes = readBytes(in, 4);
        return new String(bytes, "US-ASCII");
    }

    private static String readTagOrNull(InputStream in) throws IOException {
        byte[] bytes = new byte[4];
        int offset = 0;
        while (offset < bytes.length) {
            int read = in.read(bytes, offset, bytes.length - offset);
            if (read < 0) {
                return offset == 0 ? null : failEof();
            }
            offset += read;
        }
        return new String(bytes, "US-ASCII");
    }

    private static String failEof() throws EOFException {
        throw new EOFException("unexpected end of WAV file");
    }

    private static int readU16(InputStream in) throws IOException {
        byte[] b = readBytes(in, 2);
        return (b[0] & 0xff) | ((b[1] & 0xff) << 8);
    }

    private static int readU32(InputStream in) throws IOException {
        byte[] b = readBytes(in, 4);
        long value = (b[0] & 0xffL)
                | ((b[1] & 0xffL) << 8)
                | ((b[2] & 0xffL) << 16)
                | ((b[3] & 0xffL) << 24);
        if (value > Integer.MAX_VALUE) {
            throw new IOException("WAV chunk is too large");
        }
        return (int) value;
    }

    private static byte[] readBytes(InputStream in, int size) throws IOException {
        byte[] bytes = new byte[size];
        int offset = 0;
        while (offset < size) {
            int read = in.read(bytes, offset, size - offset);
            if (read < 0) {
                throw new EOFException("unexpected end of WAV file");
            }
            offset += read;
        }
        return bytes;
    }

    private static void skipFully(InputStream in, int size) throws IOException {
        int remaining = size;
        byte[] buffer = null;
        while (remaining > 0) {
            long skipped = in.skip(remaining);
            if (skipped > 0) {
                remaining -= (int) skipped;
                continue;
            }
            if (buffer == null) {
                buffer = new byte[4096];
            }
            int read = in.read(buffer, 0, Math.min(buffer.length, remaining));
            if (read < 0) {
                throw new EOFException("unexpected end of WAV file");
            }
            remaining -= read;
        }
    }
}
