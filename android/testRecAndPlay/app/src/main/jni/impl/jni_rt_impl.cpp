#include <jni.h>
#include <string.h>
#include <android/log.h>

#include "rt_impl.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <memory>

#ifndef LOG_TAG
#define LOG_TAG "donkey_debug"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif

static Realtime_NS* pRealtime_NS_static_pointer = 0;



//============================[  ]=====================
extern "C"
JNIEXPORT void JNICALL
Java_com_example_testRecAndPlay_RealtimeNS_initial(
        JNIEnv *env, jobject instance,
        jobject assetmanager,
        jfloat samplerate){

//    __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug", "initial start");
    LOGE("donkey_debug jni init","start initial jni pos1");
    if(pRealtime_NS_static_pointer){
        //wild pointer dirty objects
        LOGE("donkey_debug jni init","start initial jni if start");

        LOGE("donkey_debug jni init","start initial jni if end");
    }else{
        //new objects
        LOGE("donkey_debug","start initial jni pos else start");
        AAssetManager* manager = AAssetManager_fromJava(env, assetmanager);

        pRealtime_NS_static_pointer = new Realtime_NS();
        pRealtime_NS_static_pointer->initialModel(manager);





        //==================

        //==================






        LOGE("donkey_debug","start initial jni pos else end");
//        __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug", "initial new and bufferLen");
    }
}




extern "C"
JNIEXPORT void JNICALL
Java_com_example_testRecAndPlay_RealtimeNS_processBlock(
        JNIEnv *env, jobject instance,
        jshortArray buffer_s16, jint len) {

    __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug jni processBlock", "start processSamples");
    __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug jni processBlock", "input buffer len %d",len);
    // 获取 buffer_s16 的底层指针
    short* buffer = env->GetShortArrayElements(buffer_s16, NULL);
    int length = env->GetArrayLength(buffer_s16);
    __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug jni processBlock", "now length %d",length);

    pRealtime_NS_static_pointer->processBlock(buffer,len);

    // 释放 buffer_s16 的底层指针
    env->ReleaseShortArrayElements(buffer_s16, buffer, JNI_ABORT);

    __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug jni processBlock", "end processSamples");
}





