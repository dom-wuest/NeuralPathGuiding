#pragma once

#include "FalcorCUDA.h"
#include "Falcor.h"

#include "tiny-cuda-nn/common.h"
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include "json/json.hpp"

namespace tcnn {
    template<typename, typename, typename>
    class Trainer;

    template<typename>
    class Loss;

    template<typename>
    class Optimizer;

    template<typename>
    class NetworkWithInputEncoding;

    template <typename T>
    class Encoding;
};


using precision_t = tcnn::network_precision_t;


namespace Falcor
{

    class dlldecl SimpleNNModel
    {
    public:

        SimpleNNModel(std::string model_name, nlohmann::json config, const uint32_t n_input_dims, const uint32_t n_output_dims);
        SimpleNNModel(std::string model_name, std::string config_path);

        bool fit(
            Buffer::SharedPtr X, 
            Buffer::SharedPtr y,  
            const uint32_t numTrainingElements, 
            GpuFence::SharedPtr dataPreparationFence,
            uint64_t dataPreparationFenceValue,
            uint32_t numBatches=1,
            int32_t outputsize_overwrite=-1
        );
        bool predict(
            Buffer::SharedPtr X, 
            Buffer::SharedPtr y,  
            const uint32_t numPredictElements, 
            GpuFence::SharedPtr dataPreparationFence,
            GpuFence::SharedPtr predictionResultFence,
            uint64_t dataPreparationFenceValue
        );

        bool renderUI(Gui::Widgets& widget);
        bool getTrainFlag() { return train_flag; };
        void setTrainFlag(bool flag) { train_flag = flag; };

    private:
        void createNetwork(nlohmann::json config, const uint32_t n_input_dims, const uint32_t n_output_dims);
        void createNetwork(std::string configPath);

        std::shared_ptr<tcnn::Loss<precision_t>> loss;
        std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer;
        std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network;

        std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer;

        uint32_t m_input_dims;
        uint32_t m_output_dims;
        
        nlohmann::json m_config;

        tcnn::default_rng_t rng{1337};
        
        cudaStream_t inference_stream;
        cudaStream_t training_stream;

        std::string name;

        bool train_flag {false};

        bool bInitCudaFenceForTrainingData = false;
        bool bInitCudaFenceOut = false;
        bool bFirstRun = true;
        
        cudaExternalSemaphore_t cudaFenceForTrainingData;
        cudaExternalSemaphore_t cudaFenceOut;

        nlohmann::json loss_opts;
        nlohmann::json optimizer_opts;
        nlohmann::json encoding_opts;
        nlohmann::json network_opts;
        float loss_value;
        float loss_value_ema;
        uint32_t numTrainElements = 0;
        uint32_t m_batch_size;
        uint32_t numTrainSteps = 0;
        std::string m_config_path;
    };

}
