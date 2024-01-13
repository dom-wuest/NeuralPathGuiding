#include <stdafx.h>
#include "NeuralNetworkModel.h"
#include "tiny-cuda-nn/network_with_input_encoding.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"
#pragma comment(lib, "cuda")
#pragma comment(lib, "cudart")
#pragma comment(lib, "curand")
#pragma comment(lib, "cublas")

#define CUDA_CHECK_SUCCESS(x)                                                                            \
    do {                                                                                                 \
        cudaError_t result = x;                                                                          \
        if (result != cudaSuccess)                                                                       \
        {                                                                                                \
            logError("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                    \
        }                                                                                                \
    } while(0)


__global__ void eval_buff(uint32_t n_elements, uint32_t stride_X, uint32_t stride_y, uint32_t size,
    float* __restrict__ select,
    float* __restrict__ X, float* __restrict__ y,
    float* __restrict__ result_X, float* __restrict__ result_y)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    uint32_t input_idx = i * stride_X;
    uint32_t output_idx = i * stride_y;

    uint32_t selected_idx = (int)(select[i] * size);

    uint32_t selected_input_idx = selected_idx * stride_X;
    uint32_t selected_output_idx = selected_idx * stride_y;

    for (uint32_t j = 0; j < stride_X; ++j) {
        result_X[input_idx + j] = X[selected_input_idx + j];
    }

    for (uint32_t j = 0; j < stride_y; ++j) {
        result_y[output_idx + j] = y[selected_output_idx + j];
    }
}

void Falcor::SimpleNNModel::createNetwork(nlohmann::json config, const uint32_t n_input_dims, const uint32_t n_output_dims) {
    m_config = config;
    m_input_dims = n_input_dims;
    m_output_dims = n_output_dims;

    nlohmann::json loss_opts = config.value("loss", nlohmann::json::object());
    nlohmann::json optimizer_opts = config.value("optimizer", nlohmann::json::object());
    nlohmann::json encoding_opts = config.value("encoding", nlohmann::json::object());
    nlohmann::json network_opts = config.value("network", nlohmann::json::object());

    loss.reset(tcnn::create_loss<precision_t>(loss_opts));
    optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_opts));
    network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

    trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
}

Falcor::SimpleNNModel::SimpleNNModel(std::string model_name, nlohmann::json config, const uint32_t n_input_dims, const uint32_t n_output_dims)
{
    name = model_name;
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    training_stream = inference_stream;

    createNetwork(config, n_input_dims, n_output_dims);

    

}

Falcor::SimpleNNModel::SimpleNNModel(std::string model_name, std::string config_path) {
    name = model_name;
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    training_stream = inference_stream;

    createNetwork(config_path);
}

void Falcor::SimpleNNModel::createNetwork(std::string config_path) {
    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;
    if (findFileInShaderDirectories(config_path, networkConfigFullPath)) {
        networkConfigStream.open(networkConfigFullPath);
    }
    else {
        throw std::runtime_error("DummyNeuralNetwork: No network config found");
    }

    bFirstRun = true;
    numTrainSteps = 0;
    loss_value = 0;
    m_batch_size = 16384;

    nlohmann::json config;
    networkConfigStream >> config;
    networkConfigStream.close();
    m_config_path = config_path;
    createNetwork(config, config["dims"]["input"], config["dims"]["output"]);
}


namespace Falcor {

    tcnn::GPUMemory<float> map_falcor_to_cuda(Buffer::SharedPtr falcor_buffer)
    {
        tcnn::GPUMemory<float> tcnn_buffer((float*)falcor_buffer->getCUDADeviceAddress(), falcor_buffer->getSize());
        return tcnn_buffer;
    }

}

bool Falcor::SimpleNNModel::fit(
    Buffer::SharedPtr X, 
    Buffer::SharedPtr y,  
    const uint32_t numTrainingElements, 
    GpuFence::SharedPtr dataPreparationFence,
    uint64_t dataPreparationFenceValue,
    uint32_t batch_size,
    int32_t outputsize_overwrite)
{
    cudaExternalSemaphoreWaitParams extSemaphoreParams = {};
    extSemaphoreParams.params.fence.value = dataPreparationFenceValue;
    // Waiting shaders execution. They fill all structure buffers for our networks
    
    if (!bInitCudaFenceForTrainingData) {
        cudaExternalSemaphoreHandleDesc extSemaphoreDesc = {};
        extSemaphoreDesc.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
        extSemaphoreDesc.handle.win32.handle = dataPreparationFence->getSharedApiHandle();
        CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceForTrainingData, &extSemaphoreDesc));
        bInitCudaFenceForTrainingData = true;
    }
    
    CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync_v2(&cudaFenceForTrainingData, &extSemaphoreParams, 1, training_stream));


    const uint32_t train_steps = (numTrainingElements + m_batch_size - 1) / m_batch_size;
    uint32_t buffer_len = numTrainingElements/train_steps;

    tcnn::GPUMemory<float> tcnn_X = map_falcor_to_cuda(X);
    tcnn_X.setShared(true);

    tcnn::GPUMemory<float> tcnn_y = map_falcor_to_cuda(y);
    tcnn_y.setShared(true);



    uint32_t output_size = (outputsize_overwrite == -1) ? m_output_dims : outputsize_overwrite;


    uint32_t offset = 0;
    loss_value = 0.0f;
    for (uint32_t i = 0; i < train_steps; i++) {
        float l_v = 0.0f;

        uint32_t real_size = buffer_len / 128 * 128;
        tcnn::GPUMatrix<float> training_X(tcnn_X.data()+ offset* m_input_dims, m_input_dims, real_size);
        tcnn::GPUMatrix<float> training_y(tcnn_y.data()+ offset* output_size, output_size, real_size);

        // Maybe we should shuffle training data? 
        trainer->training_step(training_stream, training_X, training_y, &l_v);

        loss_value += l_v / train_steps;
        offset += buffer_len;
    }

    numTrainSteps += 1;
    numTrainElements = numTrainingElements;
    if (!bFirstRun)
        loss_value_ema = loss_value_ema * 0.99 + loss_value * 0.01;
    else
    {
        loss_value_ema = loss_value;
        bFirstRun= false;
    }
    
    return true;
}

bool Falcor::SimpleNNModel::predict(
    Buffer::SharedPtr X, 
    Buffer::SharedPtr y,  
    const uint32_t numPredictElements, 
    GpuFence::SharedPtr dataPreparationFence,
    GpuFence::SharedPtr predictionResultFence,
    uint64_t dataPreparationFenceValue)
{
    cudaExternalSemaphoreWaitParams extSemaphoreParams = {};
    extSemaphoreParams.params.fence.value = dataPreparationFenceValue;
    // Waiting shaders execution. They fill all structure buffers for our networks
    
    if (!bInitCudaFenceForTrainingData) {
        cudaExternalSemaphoreHandleDesc extSemaphoreDesc = {};
        extSemaphoreDesc.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
        extSemaphoreDesc.handle.win32.handle = dataPreparationFence->getSharedApiHandle();
        CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceForTrainingData, &extSemaphoreDesc));
        bInitCudaFenceForTrainingData = true;
    }
    
    CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync_v2(&cudaFenceForTrainingData, &extSemaphoreParams, 1, training_stream));

    uint32_t buffer_len = numPredictElements;

    tcnn::GPUMemory<float> tcnn_X = map_falcor_to_cuda(X);
    tcnn_X.setShared(true);
    tcnn::GPUMemory<float> tcnn_y = map_falcor_to_cuda(y);
    tcnn_y.setShared(true);
    tcnn::GPUMatrix<float> inference_X(tcnn_X.data(), m_input_dims, buffer_len);

    tcnn::GPUMatrix<float> inference_y(tcnn_y.data(), m_output_dims, buffer_len);

    network->inference(inference_stream, inference_X, inference_y);

    // Sync with Rendering backend. 
    if (!bInitCudaFenceOut) {
        cudaExternalSemaphoreHandleDesc extSemaphoreDesc = {};
        extSemaphoreDesc.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
        extSemaphoreDesc.handle.win32.handle = predictionResultFence->getSharedApiHandle();
        CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceOut, &extSemaphoreDesc));
        bInitCudaFenceOut = true;
    }

    cudaExternalSemaphoreSignalParams extSemaphoreParamsOut = {};
    extSemaphoreParamsOut.params.fence.value = predictionResultFence->beforeExternalSignal();
    CUDA_CHECK_THROW(cudaSignalExternalSemaphoresAsync_v2(&cudaFenceOut, &extSemaphoreParamsOut, 1, training_stream));
    
    return true;
}

bool Falcor::SimpleNNModel::renderUI(Gui::Widgets& widget)
{

    if (auto nnGroup = widget.group(name, true))
    {
        nnGroup.text("Loss: " + std::to_string(loss_value));
        nnGroup.text("EMA Loss: " + std::to_string(loss_value_ema));
        nnGroup.text("Num Train Steps: " + std::to_string(numTrainSteps));
        nnGroup.text("Num Train Elements: " + std::to_string(numTrainElements));

  
        if (nnGroup.var("Batch Size", m_batch_size, (uint32_t)128, (uint32_t)128000, 128))
        {
            m_batch_size = (m_batch_size + 127) / 128 * 128;
        }
        nnGroup.checkbox("Training", train_flag);

        if (nnGroup.button("Reset Weights", false))
        {
            createNetwork(m_config, m_input_dims, m_output_dims);
        }

        if (!m_config_path.empty()) {
            if (nnGroup.button("Reload model", false))
            {
                createNetwork(m_config_path);
            }
        }
            
    }

    return false;
}
