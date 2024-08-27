
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Tensor.hpp>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <vector>

class MNNModel{
public:
    MNNModel(char* buf, size_t buf_size, bool use_gpu, const char *input_tensor_name);
    MNNModel(const char* mnn_name, bool use_gpu, const char *input_tensor_name);
    void inference(std::vector<float>& inputs, std::vector<float>& calc_480, const char *output_tensor_name);
    ~MNNModel();
private:
    MNN::Tensor *_input;
    MNN::Session *_session;
    std::shared_ptr<MNN::Interpreter> _interpreter;
};




