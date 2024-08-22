#pragma once

#include <memory>
#include <vector>
#include <android/log.h>
#include "../base/audio_ring_buffer.h"

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
    Realtime_NS(){
        __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug Realtime_NS", "start construction");

        pInBuffer_960.reset(new AudioRingBuffer(1,960));
        pOutBuffer_960.reset(new AudioRingBuffer(1,960));
        std::vector<float> mEmpty(480);
        pOutBuffer_960->Write(mEmpty,1,480);

        mFloat32_160.resize(160);
        mFloat32_480.resize(480);
        mPorcessBuf_960.resize(960);
        mCalc_480.resize(480);

        __android_log_print(ANDROID_LOG_DEBUG, "donkey_debug Realtime_NS", "end construction");
    }

    ~Realtime_NS(){}


    void processBlock(int16_t* buffer, int len){
        assert(len >0);

        //int16 to float32 transfer
        for(int idx=0;idx<len;idx++){
            mFloat32_160.data()[idx] = S16ToFloat(buffer[idx]);
        }

        //======================
        //ring buffer push data
        if(pInBuffer_960->WriteFramesAvailable() >= 160){
            pInBuffer_960->Write(mFloat32_160,1,160);
        }

        //=======================

        while(pInBuffer_960->ReadFramesAvailable()>=480){
            //process once
            pInBuffer_960->Read(mFloat32_480,1,480);
            std::copy(mPorcessBuf_960.begin()+480,mPorcessBuf_960.end(),mPorcessBuf_960.begin());
            std::copy(mFloat32_480.begin(),mFloat32_480.end(),mPorcessBuf_960.begin()+480);

            //=====================
            //process model in here
            std::copy(mPorcessBuf_960.begin(),mPorcessBuf_960.begin()+480,mCalc_480.begin());
            //====================

            if(pOutBuffer_960->WriteFramesAvailable()>=480){
                pOutBuffer_960->Write(mCalc_480,1,480);
            }
        }

        //======================
        //ring buffer get data
        if(pOutBuffer_960->ReadFramesAvailable()>=160){
            pOutBuffer_960->Read(mFloat32_160,1,160);
        }

        for(int idx=0;idx<len;idx++){
            buffer[idx] = FloatToS16(mFloat32_160.data()[idx]);
        }

    }

private:
    std::unique_ptr<AudioRingBuffer> pInBuffer_960 = nullptr;
    std::unique_ptr<AudioRingBuffer> pOutBuffer_960 = nullptr;
    std::vector<float> mFloat32_160;
    std::vector<float> mFloat32_480;
    std::vector<float> mPorcessBuf_960;
    std::vector<float> mCalc_480;

};