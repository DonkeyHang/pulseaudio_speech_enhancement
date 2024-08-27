#pragma once

#include <memory>
#include <vector>
#include <android/log.h>
#include "../base/include/audio_ring_buffer.h"
#include <android/asset_manager.h>
#include "../base/include/rtns_model.h"

#ifndef LOG_TAG
#define LOG_TAG "donkey_debug"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


typedef std::numeric_limits<int16_t> limits_int16;


static inline float S16ToFloat(int16_t v) {

    static const float kMaxInt16Inverse = 1.f / limits_int16::max();
    static const float kMinInt16Inverse = 1.f / limits_int16::min();
    return v * (v > 0 ? kMaxInt16Inverse : -kMinInt16Inverse);
}

static inline int16_t FloatToS16(float v) {
    if (v > 0)
        return v >= 1 ? limits_int16::max()
                      : static_cast<int16_t>(v * limits_int16::max() + 0.5f);
    return v <= -1 ? limits_int16::min()
                   : static_cast<int16_t>(-v * limits_int16::min() - 0.5f);
}


class Realtime_NS{
public:
    Realtime_NS();
    ~Realtime_NS();

    void initialModel(AAssetManager* manager);

    void processBlock(int16_t* buffer, int len);


private:
    std::unique_ptr<AudioRingBuffer> pInBuffer_960 = nullptr;
    std::unique_ptr<AudioRingBuffer> pOutBuffer_960 = nullptr;
    std::vector<float> mFloat32_160;
    std::vector<float> mFloat32_480;
    std::vector<float> mPorcessBuf_960;
    std::vector<float> mCalc_480;

    std::unique_ptr<MNNModel> pModel = nullptr;

};