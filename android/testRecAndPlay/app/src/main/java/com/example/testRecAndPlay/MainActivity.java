package com.example.testRecAndPlay;

//=================================================

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.support.v7.app.AlertDialog;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v4.content.ContextCompat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import android.content.res.AssetManager;


//=================================================
public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1;

    private boolean isRecording = false;

    private AudioRecord audioRecord;
    private AudioTrack audioTrack;
    private int bufferSize;

    private RealtimeNS rt_ns;

    private AssetManager assetmanager;

    private float[] hammingWindow48to16;  // Window for downsample from 48k to 16k
    private float[] hammingWindow16to48;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.e("donkey_debug", "start oncreate: " );
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button startButton = findViewById(R.id.start_button);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            }
        });

        initHammingWindows();

        assetmanager = getAssets();

        rt_ns = new RealtimeNS();
        rt_ns.debugxxxx();
        rt_ns.initial(assetmanager,48000);
//        rt_ns.initial(48000);



        //=============================================================
        // 检查并申请录音权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            // 录音权限已授予
            Toast.makeText(this, "录音权限已授予", Toast.LENGTH_SHORT).show();
        } else {
            // 显示权限解释对话框
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.RECORD_AUDIO)) {
                new AlertDialog.Builder(this)
                        .setTitle("权限申请")
                        .setMessage("需要录音权限才能正常工作")
                        .setPositiveButton("允许", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSION_REQUEST_CODE);
                            }
                        })
                        .setNegativeButton("拒绝", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                dialogInterface.dismiss();
                            }
                        })
                        .show();
            } else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSION_REQUEST_CODE);
            }
        }
    }

    private void initHammingWindows() {
        // Assuming these ratios, you may calculate it dynamically based on your actual sample rates
        hammingWindow48to16 = calculateHammingWindow(3); // Example ratio
        hammingWindow16to48 = calculateHammingWindow(3); // Example ratio
    }

    private float[] calculateHammingWindow(int size) {
        float[] window = new float[size];
        for (int i = 0; i < size; i++) {
            window[i] = (float) (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (size - 1)));
        }
        return window;
    }

    private short[] convertByteArrayToShortArray(byte[] byteArray) {
        short[] shortArray = new short[byteArray.length / 2];
        ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shortArray);
        return shortArray;
    }

    private byte[] convertShortArrayToByteArray(short[] shortArray) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(shortArray.length * 2);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (short value : shortArray) {
            byteBuffer.putShort(value);
        }
        return byteBuffer.array();
    }

    private void startRecording() {
        bufferSize = AudioRecord.getMinBufferSize(48000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, 48000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize);

        final byte[] buffer = new byte[bufferSize];
        audioRecord.startRecording();
        isRecording = true;

        audioTrack = new AudioTrack(android.media.AudioManager.STREAM_MUSIC, 48000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize, AudioTrack.MODE_STREAM);
        audioTrack.play();

        new Thread(new Runnable() {
            @Override
            public void run() {
                while (isRecording) {
                    int bytesRead = audioRecord.read(buffer, 0, bufferSize);
                    short[] shortBuffer = convertByteArrayToShortArray(buffer);
                    short[] downsampled_16k = downsample(shortBuffer, 48000, 16000);

                    //===================================
                    // Process block would be here
                    rt_ns.processBlock(downsampled_16k,160);
                    //===================================

                    short[] upsampled_48k = upsample(downsampled_16k, 16000, 48000);
                    byte[] byteBuffer = convertShortArrayToByteArray(upsampled_48k);
                    audioTrack.write(byteBuffer, 0, bytesRead);
                }
            }
        }).start();
    }

    public short[] downsample(short[] input, int originalRate, int targetRate) {
        int ratio = originalRate / targetRate;
//        float[] window = hammingWindow(ratio);
//        float[] window = hammingWindow48to16;
        int outputLength = input.length / ratio;
        short[] output = new short[outputLength];

        for (int index = 0; index < outputLength; index++) {
            double sum = 0.0;
            for (int i = 0; i < ratio; i++) {
                int inputIndex = index * ratio + i;
                if (inputIndex < input.length) {
                    sum += input[inputIndex] * hammingWindow48to16[i];
                }
            }
            output[index] = (short) ((sum / ratio));
        }
        return output;
    }

    public short[] upsample(short[] input, int originalRate, int targetRate) {
        int ratio = targetRate / originalRate;
        short[] result = new short[input.length * ratio];
//        float[] window = hammingWindow(ratio);

        for (int i = 0; i < input.length; i++) {
            result[i * ratio] = input[i];
            if (i < input.length - 1) {
                short nextSample = input[i + 1];
                for (int k = 1; k < ratio; k++) {
                    float interpolatedValue = input[i] + (nextSample - input[i]) * (float) k / ratio;
                    result[i * ratio + k] = (short) ((interpolatedValue * hammingWindow16to48[k]));
                }
            }
        }
        return result;
    }

    private void stopRecording() {
        isRecording = false;
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        if (audioTrack != null) {
            audioTrack.stop();
            audioTrack.release();
            audioTrack = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // 录音权限已授予
                Toast.makeText(this, "录音权限已授予", Toast.LENGTH_SHORT).show();
            } else {
                // 录音权限被拒绝
                Toast.makeText(this, "录音权限被拒绝", Toast.LENGTH_SHORT).show();
            }
        }
    }











}