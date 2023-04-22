package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;


import com.example.myapplication.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.util.Log;

public class MainActivity extends AppCompatActivity {
    private final static String TAG = "MainActivity";
    private void copyBigDataToSD(String strOutFileName) throws IOException {
        Log.i(TAG, "start copy file " + strOutFileName);
        File sdDir = getExternalFilesDir(null);//get root dir
        File file = new File(sdDir.toString()+"/models/");
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString()+"/models/" + strOutFileName;
        File f = new File(tmpFile);
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/models/"+ strOutFileName);
        myInput = this.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
        Log.i(TAG, "end copy file " + strOutFileName);

    }

    private byte[] convertToBgrByteArray(String imagePath, int [] imgSize) {
        // Load the image from file
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);

        // Get the width and height of the image
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        imgSize[0] = width;
        imgSize[1] = height;
        // Create a byte array for the bgr image
        byte[] bgrImage = new byte[width * height * 3];

        // Loop through each pixel of the image and get the RGB values
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = bitmap.getPixel(x, y);
                int red = Color.red(pixel);
                int green = Color.green(pixel);
                int blue = Color.blue(pixel);
                bgrImage[index++] = (byte) blue;
                bgrImage[index++] = (byte) green;
                bgrImage[index++] = (byte) red;
            }
        }
        return bgrImage;
    }

    private boolean initFaceEngine(String rootModelPath){
        Log.i(TAG, "Init detection model...");
        boolean isDetectionInitOk = faceDetectionInit(rootModelPath);
        if(isDetectionInitOk){
            Log.i(TAG, "Init detection model successfully");
        }
        else {
            Log.i(TAG, "Init detection model failed");
            return false;
        }
        boolean isVectorizationInitOk = faceVectorizationInit(rootModelPath);
        if(isVectorizationInitOk){
            Log.i(TAG, "Init vectorization model successfully");
        }
        else {
            Log.i(TAG, "Init vectorization model failed");
            return false;
        }
        boolean isAlignmentInitOk = faceAlignmentInit();
        if(isAlignmentInitOk){
            Log.i(TAG, "Init alignment model successfully");
        }
        else {
            Log.i(TAG, "Init alignment model failed");
            return false;
        }
        return true;
    }

    private void extractFaceInfo(float[] faceDetectionRes,
                                      float[][] features,
                                      float[][] faceBoxes,
                                      int numFace){
        for (int faceIdx=0; faceIdx<numFace; faceIdx++){
            for (int i=0; i<4; i++){
                faceBoxes[faceIdx][i] = faceDetectionRes[526 * faceIdx + i + 1];
            }
            for (int j=0; j<512; j++){
                features[faceIdx][j] = faceDetectionRes[526 * faceIdx + j + 15];
            }
        }
    }

    // Used to load the 'myapplication' library on application startup.
    static {
        System.loadLibrary("face_recognition_mnn");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        //copy model
        try {
            copyBigDataToSD("scrfd_2.5g_bnkps_shape320x320.mnn");
            copyBigDataToSD("glint360k_cosface.mnn");
            copyBigDataToSD("aligned_face_id2.jpg");
            copyBigDataToSD("aligned_face.jpg");
            copyBigDataToSD("04_14_211128939.jpeg");
            copyBigDataToSD("test2face.jpg");
            copyBigDataToSD("326415.jpeg");
        } catch (IOException e) {
            e.printStackTrace();
        }
        File sdDir = getExternalFilesDir(null);//get model store dir
        String sdPath = sdDir.toString() + "/models/";
        boolean initFaceEngineOk = initFaceEngine(sdPath);
        if(initFaceEngineOk){
            int [] sizeImg1 = {0, 0};
            byte[] inputImage1 = convertToBgrByteArray(sdPath+"aligned_face_id2.jpg", sizeImg1);
            int width1 = sizeImg1[0];
            int height1 = sizeImg1[1];

            int [] sizeImg2 = {0, 0};
            byte[] inputImage2 = convertToBgrByteArray(sdPath+"aligned_face_id2.jpg", sizeImg2);
            int width2 = sizeImg1[0];
            int height2 = sizeImg1[1];

            float [] results1 = extractFeaturePipeline(inputImage1, width1, height1);
            float [] results2 = extractFeaturePipeline(inputImage2, width2, height2);
            if(results1[0]>0 && results2[0]>0) {
                int numFace1 = (int) results1[0];
                float[][] outFaceBoxes1 = new float[numFace1][4];
                float[][] outFaceVector1 = new float[numFace1][512];
                extractFaceInfo(results1, outFaceVector1, outFaceBoxes1, numFace1);
                int numFace2 = (int) results2[0];
                float[][] outFaceBoxes2 = new float[numFace2][4];
                float[][] outFaceVector2 = new float[numFace2][512];
                extractFaceInfo(results2, outFaceVector2, outFaceBoxes2, numFace2);
                float score = calculateSimilarity(outFaceVector1[0], outFaceVector2[0]);
                Log.i(TAG, "score = " + score);
            }

        }

        tv.setText(stringFromJNI(sdPath));
    }

    /**
     * A native method that is implemented by the 'myapplication' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI(String model_path);
    public native boolean faceDetectionInit(String modelPath);
    public native float[] faceDetect(byte [] imageData, int with , int height);
    public native boolean faceVectorizationInit(String modelPath);
    public native float[] faceVectorize(byte [] imageData);
    public native boolean faceAlignmentInit();
    public native float[] extractFeaturePipeline(byte [] imageData, int with , int height);
    public native float calculateSimilarity(float [] feat1, float [] feat2);
}