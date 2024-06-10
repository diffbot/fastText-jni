package com.diffbot.fasttext;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Files;

public class FastTextModel implements AutoCloseable {

    public FastTextModel(ByteBuffer byteBuffer) {
        if (!byteBuffer.isDirect()) {
            throw new RuntimeException("ByteBuffer must be direct");
        }
        this.load(byteBuffer);
    }

    public FastTextModel(InputStream inputStream) throws IOException {
        byte[] bytes = inputStream.readAllBytes();
        ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length);
        buffer.put(bytes);
        buffer.flip();
        this.load(buffer);
    }

    private native void load(ByteBuffer byteBuffer);
    public native Prediction predictProba(String s);
    public native Prediction[] predictProbaTopK(String s, int k);
    public native Prediction[] predictProbaWithThreshold(String s, float threshold);

    @Override
    public native void close();

    @SuppressWarnings("unused")
    private long handle;

    private static final String[] LIBRARIES_TO_LOAD = { "libfasttext" };
    private static boolean loadedLibraries = false;

    // Load library based on OS
    static{
        // Find platform
        String OS = "linux";
        String extension = ".so";
        String rawOS = System.getProperty("os.name", "generic").toLowerCase();
        if ((rawOS.contains("mac")) || (rawOS.contains("darwin"))) {
            String arch = System.getProperty("os.arch");
            OS = "mac";
            extension = "-mac-" + arch + ".so";
        }else if(rawOS.contains("win")){
            OS = "windows";
        }

        int loaded = 0;
        for(String name : LIBRARIES_TO_LOAD) {
            String resource = name + extension;
            URL libraryResource = FastTextModel.class.getResource(resource);
            File tmpDir = null;
            File nativeLibTmpFile = null;
            try {
                tmpDir = Files.createTempDirectory(name).toFile();

                nativeLibTmpFile = new File(tmpDir, name);
                try (InputStream in = libraryResource.openStream()) {
                    Files.copy(in, nativeLibTmpFile.toPath());
                }
                System.load(nativeLibTmpFile.getAbsolutePath());
                loaded++;
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (nativeLibTmpFile != null) {
                    nativeLibTmpFile.delete();
                }
                if (tmpDir != null) {
                    tmpDir.delete();
                }
            }
        }

        if(loaded == LIBRARIES_TO_LOAD.length){
            loadedLibraries = true;
        }else{
            System.err.println("Failed to load native libraries");
        }
    }
}
