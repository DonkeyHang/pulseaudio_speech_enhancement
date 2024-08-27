#include "rt_impl.h"



Realtime_NS::Realtime_NS(){
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

Realtime_NS::~Realtime_NS(){

}

void Realtime_NS::initialModel(AAssetManager* manager){
    AAsset* asset = AAssetManager_open(manager, "model.mnn", AASSET_MODE_BUFFER);

    int bufferSize = AAsset_getLength(asset);
    LOGE("file size:%d",bufferSize);

    char* pBuf = new char[bufferSize];
    memset(pBuf, 0x00, bufferSize);
    AAsset_read(asset, pBuf, bufferSize);

    pModel.reset(new MNNModel(pBuf,bufferSize,true,"input"));

    delete[] pBuf;
    AAsset_close(asset);
}

void Realtime_NS::processBlock(int16_t* buffer, int len){
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
//        std::copy(mPorcessBuf_960.begin(),mPorcessBuf_960.begin()+480,mCalc_480.begin());
        pModel->inference(mPorcessBuf_960,mCalc_480,"output");

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
