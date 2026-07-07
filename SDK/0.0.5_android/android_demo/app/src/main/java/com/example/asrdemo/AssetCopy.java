package com.example.asrdemo;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public final class AssetCopy {
    private AssetCopy() {}

    public static void copyTree(Context context, String assetPath, File outDir)
            throws IOException {
        AssetManager assets = context.getAssets();
        String[] children = assets.list(assetPath);
        if (children == null || children.length == 0) {
            copyFile(assets, assetPath, outDir);
            return;
        }
        if (!outDir.exists() && !outDir.mkdirs()) {
            throw new IOException("failed to create " + outDir);
        }
        for (String child : children) {
            String childAsset = assetPath.isEmpty() ? child : assetPath + "/" + child;
            copyTree(context, childAsset, new File(outDir, child));
        }
    }

    public static void copyFile(Context context, String assetPath, File outFile)
            throws IOException {
        copyFile(context.getAssets(), assetPath, outFile);
    }

    private static void copyFile(AssetManager assets, String assetPath, File outFile)
            throws IOException {
        File parent = outFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create " + parent);
        }
        try (InputStream in = assets.open(assetPath);
             OutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[1024 * 64];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }
}
