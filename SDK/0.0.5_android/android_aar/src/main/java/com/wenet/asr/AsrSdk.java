package com.wenet.asr;

import android.content.Context;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class AsrSdk {
    private static final String MODEL_DIR_NAME = "asr_model";
    private static final String MODEL_TMP_DIR_NAME = "asr_model.tmp";
    private static final String MODEL_MANIFEST_NAME = "sdk_model.json";

    private AsrSdk() {}

    public static File installModelZip(Context context, File modelZip) throws IOException {
        try (InputStream in = new FileInputStream(modelZip)) {
            return installModelZip(context, in);
        }
    }

    public static File installModelZip(Context context, InputStream modelZip) throws IOException {
        File finalDir = getDefaultModelDir(context);
        File tmpDir = new File(context.getFilesDir(), MODEL_TMP_DIR_NAME);
        deleteTree(tmpDir);
        if (!tmpDir.mkdirs()) {
            throw new IOException("failed to create " + tmpDir);
        }

        extractZip(modelZip, tmpDir);
        File manifest = new File(tmpDir, MODEL_MANIFEST_NAME);
        if (!manifest.isFile()) {
            deleteTree(tmpDir);
            throw new IOException("model zip is missing " + MODEL_MANIFEST_NAME);
        }

        deleteTree(finalDir);
        if (!tmpDir.renameTo(finalDir)) {
            deleteTree(tmpDir);
            throw new IOException("failed to install model to " + finalDir);
        }
        return finalDir;
    }

    public static boolean isModelInstalled(Context context) {
        return new File(getDefaultModelDir(context), MODEL_MANIFEST_NAME).isFile();
    }

    public static File getDefaultModelDir(Context context) {
        return new File(context.getFilesDir(), MODEL_DIR_NAME);
    }

    public static AsrRecognizer createRecognizer(Context context) {
        return createRecognizer(getDefaultModelDir(context));
    }

    public static AsrRecognizer createRecognizer(File modelDir) {
        return new AsrRecognizer(modelDir);
    }

    public static String version() {
        return AsrNative.version();
    }

    public static int abiVersion() {
        return AsrNative.abiVersion();
    }

    public static String buildInfoJson() {
        return AsrNative.buildInfoJson();
    }

    private static void extractZip(InputStream input, File outDir) throws IOException {
        String outRoot = outDir.getCanonicalPath() + File.separator;
        try (ZipInputStream zip = new ZipInputStream(input)) {
            ZipEntry entry;
            byte[] buffer = new byte[1024 * 64];
            while ((entry = zip.getNextEntry()) != null) {
                File outFile = new File(outDir, entry.getName());
                String outPath = outFile.getCanonicalPath();
                if (!outPath.equals(outDir.getCanonicalPath()) && !outPath.startsWith(outRoot)) {
                    throw new IOException("unsafe zip entry: " + entry.getName());
                }
                if (entry.isDirectory()) {
                    if (!outFile.exists() && !outFile.mkdirs()) {
                        throw new IOException("failed to create " + outFile);
                    }
                    continue;
                }
                File parent = outFile.getParentFile();
                if (parent != null && !parent.exists() && !parent.mkdirs()) {
                    throw new IOException("failed to create " + parent);
                }
                try (OutputStream out = new FileOutputStream(outFile)) {
                    int read;
                    while ((read = zip.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                }
            }
        }
    }

    private static void deleteTree(File file) throws IOException {
        if (!file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteTree(child);
                }
            }
        }
        if (!file.delete()) {
            throw new IOException("failed to delete " + file);
        }
    }
}
