package com.example.testRecAndPlay;

import android.util.Log;

public class RealtimeNS {
    static{
        System.loadLibrary("rt_ns");
    }

    public static native void initial(float sr);

    public static native void processBlock(short[] buffer_s16, int len);

    public void debugxxxx(){Log.e("donkey_debug","into rt_ns java");}

}
