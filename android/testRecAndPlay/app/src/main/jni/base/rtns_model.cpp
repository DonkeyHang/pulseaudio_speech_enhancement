
#include "include/rtns_model.h"

MNNModel::MNNModel(char *buf, size_t buf_size, bool use_gpu, const char *input_tensor_name) {
    _interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(buf, buf_size));
    MNN::ScheduleConfig config;
    config.numThread = 8;
    if(use_gpu) {
        config.type = MNN_FORWARD_OPENCL;
    }
    config.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;  // 内存
    backendConfig.power = MNN::BackendConfig::Power_Normal;  // 功耗
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    config.backendConfig = &backendConfig;

    _session = _interpreter->createSession(config);
    _input = _interpreter->getSessionInput(_session, input_tensor_name);
}

MNNModel::MNNModel(const char* mnn_name, bool use_gpu, const char *input_tensor_name) {
    _interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_name));
    MNN::ScheduleConfig config;
    config.numThread = 8;
    if(use_gpu) {
        config.type = MNN_FORWARD_OPENCL;
    }
    config.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;  // 内存
    backendConfig.power = MNN::BackendConfig::Power_Normal;  // 功耗
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    config.backendConfig = &backendConfig;

    _session = _interpreter->createSession(config);
    _input = _interpreter->getSessionInput(_session, input_tensor_name);
}

MNNModel::~MNNModel() {
    if(_interpreter != nullptr && _session != nullptr) {
        _interpreter->releaseModel();
        _interpreter->releaseSession(_session);
    }
}

void MNNModel::inference(std::vector<float> &inputs, std::vector<float> &calc_480, const char *output_tensor_name) {
    auto tmpTensor = new MNN::Tensor(_input, MNN::Tensor::CAFFE);
    if(inputs.size() == tmpTensor->elementSize()) {
        for(int i=0; i<tmpTensor->elementSize(); i++) {
            tmpTensor->host<float>()[i] = inputs[i];
        }
    }
    _input->copyFromHostTensor(tmpTensor);
    delete tmpTensor;
    // run session
    _interpreter->runSession(_session);
    // get output
    auto preds = _interpreter->getSessionOutput(_session, output_tensor_name);
    MNN::Tensor hostPreds(preds, preds->getDimensionType());
    preds->copyToHostTensor(&hostPreds);

    for(int idx=0;idx<480;idx++){
        calc_480.data()[idx] = hostPreds.host<float>()[idx];
    }

//    // return vector
//    std::vector<float> result;
//    result.reserve(hostPreds.elementSize());
//    for(int i=0; i<hostPreds.elementSize(); i++) {
//        result.push_back(hostPreds.host<float>()[i]);
//    }
//    return result;
}
