/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/


#include "stdafx.h"

#include "NeuralNetwork.h"


#include "json/json.hpp"
#include <sstream>

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float* __restrict__ xs_and_ys, float* __restrict__ result) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    uint32_t output_idx = i * stride;
    uint32_t input_idx = i * 2;

    float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
    result[output_idx + 0] = val.x;
    result[output_idx + 1] = val.y;
    result[output_idx + 2] = val.z;

    for (uint32_t i = 3; i < stride; ++i) {
        result[output_idx + i] = 1;
    }
}
using precision_t = tcnn::network_precision_t;
// using namespace nlohmann;
namespace Falcor
{
    namespace
    {

        // Render pass output channels.
        const std::string kTrainInput = "trainInput";
        const std::string kTrainOutput = "trainOutput";
        const std::string kPredictInput = "predictInput";
        const std::string kPredictOutput = "predictOutput";

        const Falcor::ChannelList kInputChannels =
        {
            { kTrainInput,        "gTrainBuffer",       "Buffer filled with train data",                false,      ResourceFormat::Unknown },
            { kPredictInput,      "gPredictBuffer",     "BUffer filled with data for prediction",       false,      ResourceFormat::Unknown },
        };
        
        const char kParams[] = "params";
    };

    static_assert(sizeof(NeuralNetworkParams) % 16 == 0, "NeuralNetworkParams size should be a multiple of 16");

    
    NeuralNetwork::NeuralNetwork(const Dictionary& dict, const ChannelList& outputs)
        : mOutputChannels(outputs)
    {
        parseDictionary(dict);
        validateParameters();

        mInputChannels = kInputChannels ;

    }

    void NeuralNetwork::parseDictionary(const Dictionary& dict)
    {
        for (const auto& [key, value] : dict)
        {
            if (key == kParams) mSharedParams = value;
        }
    }

    Dictionary NeuralNetwork::getScriptingDictionary()
    {
        Dictionary d;
        d[kParams] = mSharedParams;
        return d;
    }

    RenderPassReflection NeuralNetwork::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        


        return reflector;
    }

    void NeuralNetwork::compile(RenderContext* pRenderContext, const CompileData& compileData)
    {
    }

    void NeuralNetwork::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        dirty |= widget.var("Neurons", mSharedParams.Neurons, 1u, 1u << 16, 1);
        dirty |= widget.var("Layers", mSharedParams.Layers, 1u, 1u << 16, 1);
        dirty |= widget.var("BatchSize", mSharedParams.BatchSize, 1u, 1u << 16, 1);
        

        // If rendering options that modify the output have changed, set flag to indicate that.
        // In execute() we will pass the flag to other passes for reset of temporal data etc.
        if (dirty)
        {
            validateParameters();
            mOptionsChanged = true;
        }

        
    }

    void NeuralNetwork::validateParameters()
    {
        if (mSharedParams.Neurons % 16  != 0 || mSharedParams.Neurons > 64)
        {
            logError("Unsupported number of neurons. It's should be divided by 16 and be less than 64 and not zero");
            mSharedParams.Neurons = (mSharedParams.Neurons / 16) * 16;
            mSharedParams.Neurons = clamp(mSharedParams.Neurons, uint(16), uint(64));
            mRecreateNN = true;
            recreateVars();
        }

        if (mSharedParams.FinalNeurons > 64 || mSharedParams.FinalNeurons < 1)
        {
            logError("Unsupported number of final neurons. It's should be be less than 64 and not zero");
            mSharedParams.FinalNeurons = clamp(mSharedParams.Neurons, uint(1), uint(64));
            mRecreateNN = true;
            recreateVars();
        }

       /* if (mSharedParams.BatchSize % 16 != 0 || mSharedParams.BatchSize > 128)
        {
            logError("Unsupported size of batch. It's should be divided by 16 and be less than 128 and not zero");
            mSharedParams.BatchSize = (mSharedParams.BatchSize / 16) * 16 ;
            mSharedParams.BatchSize = clamp(mSharedParams.BatchSize, uint(16), uint(128));
            mRecreateNN = true;
            recreateVars();
        }*/

        if (mSharedParams.Layers > 10 && mSharedParams.Layers <= 0)
        {
            logError("Unsupported number of layers. It's should be bigger than 0 and less than 10");
            mSharedParams.Layers = clamp(mSharedParams.Layers, uint(1), uint(10));
            mRecreateNN = true;
            recreateVars();
        }
    }

    
    bool NeuralNetwork::runNN(RenderContext* pRenderContext, Buffer::SharedPtr TrainXBuffer, Buffer::SharedPtr TrainYBuffer, Buffer::SharedPtr PredictXBuffer, Buffer::SharedPtr PredictOutput)
    {
        // Initializing neural network

        uint32_t TrainXSamples = TrainXBuffer->getElementCount();
        uint32_t NumInputFeaturesNew =  TrainXBuffer->getStructSize()/sizeof(float);
        uint32_t TrainYSamples = TrainYBuffer->getElementCount();
        
        uint32_t NumOutputFeaturesNew = TrainYBuffer->getStructSize() / sizeof(float);

        uint32_t PredictXSamples = PredictXBuffer->getElementCount();

        assert(TrainXSamples == TrainYSamples);
        assert(PredictXBuffer->getStructSize() / sizeof(float) == NumInputFeatures);

        // 1. TODO: Neew to configure initializatio of NN
        if (trainer == nullptr || mRecreateNN || NumInputFeatures != NumInputFeaturesNew || NumOutputFeatures != NumOutputFeaturesNew) {
            mRecreateNN = false;
            cudaDeviceProp props;
            cudaError_t error = cudaGetDeviceProperties(&props, 0);
            NumInputFeatures = NumInputFeaturesNew;
            NumOutputFeatures = NumOutputFeaturesNew;

            if (!((props.major * 10 + props.minor) >= 75)) {
                std::cout << "Turing Tensor Core operations must be run on a machine with compute capability at least 75." << std::endl;
                exit(0);
            }

            nlohmann::json config = {
               {"loss", {
                   {"otype", "RelativeL2"}
               }},
               {"optimizer", {
                   {"otype", "Adam"},
                   {"learning_rate", 1e-2},
                   {"beta1", 0.9f},
                   {"beta2", 0.99f},
                   {"l2_reg", 0.0f},
                   // The following parameters are only used when the optimizer is "Shampoo".
                   {"beta3", 0.9f},
                   {"beta_shampoo", 0.0f},
                   {"identity", 0.0001f},
                   {"cg_on_momentum", false},
                   {"frobenius_normalization", true},
               }},
               {"encoding", {
                   {"otype", "Frequency"},
                   {"n_frequencies", 12},
               }},
               {"network", {
                   {"otype", "FullyFusedMLP"},
                   {"n_neurons", mSharedParams.Neurons},
                   {"n_layers", mSharedParams.Layers},
                   {"activation", "ReLU"},
                   {"output_activation", "None"},
               }},
            };

          

            // Fourth step: train the model by sampling the above image and optimizing an error metric

            // Various constants for the network and optimization
           
            // Input & corresponding RNG
            CURAND_CHECK_THROW(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK_THROW(curandSetPseudoRandomGeneratorSeed(rng, 1337ULL));
            CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
            training_stream = inference_stream;
            CURAND_CHECK_THROW(curandSetStream(rng, training_stream));

            // Auxiliary matrices for training
            tcnn::GPUMatrix<float> training_target(NumOutputFeatures, mSharedParams.BatchSize);
            tcnn::GPUMatrix<float> training_batch(NumInputFeatures, mSharedParams.BatchSize);

            // Auxiliary matrices for evaluation
            tcnn::GPUMatrix<float> prediction(NumOutputFeatures, n_coords_padded);
            tcnn::GPUMatrix<float> inference_batch(xs_and_ys.data(), NumInputFeatures, n_coords_padded);

            nlohmann::json encoding_opts = config.value("encoding", nlohmann::json::object());
            nlohmann::json loss_opts = config.value("loss", nlohmann::json::object());
            nlohmann::json optimizer_opts = config.value("optimizer", nlohmann::json::object());
            nlohmann::json network_opts = config.value("network", nlohmann::json::object());

            std::shared_ptr<tcnn::Loss<precision_t>> loss{ tcnn::create_loss<precision_t>(loss_opts) };
            std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{ tcnn::create_optimizer<precision_t>(optimizer_opts) };
            std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(NumInputFeatures, 0, NumOutputFeatures, encoding_opts, network_opts);

            trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
        }

        /* 2 TODO. Needs to init (maybe only 1 time) and copy\retranslate memory from InputTraining buffer, OutputTraining buffer to Training Matrices
        * It's important to take into account some examples of mapping memory from Falcor to Cuda (see \Falcor\Source\Samples\CudaInterop project). But I think that this pass should get as input and output
        * buffers, not textures.. 
        */

        {
            // Dump final image if a name was specified
         //TODO:

            int width, height;
            tcnn::GPUMemory<float> image;

            // Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypePitch2D;
            resDesc.res.pitch2D.devPtr = image.data();
            resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            resDesc.res.pitch2D.width = width;
            resDesc.res.pitch2D.height = height;
            resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.normalizedCoords = true;
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.addressMode[2] = cudaAddressModeClamp;

            cudaResourceViewDesc viewDesc;
            memset(&viewDesc, 0, sizeof(viewDesc));
            viewDesc.format = cudaResViewFormatFloat4;
            viewDesc.width = width;
            viewDesc.height = height;

            cudaTextureObject_t texture;
            CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, &viewDesc));

            // Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
            //             function will be eventually possible.

            int sampling_width = width;
            int sampling_height = height;

            // Uncomment to fix the resolution of the training task independent of input image
            // int sampling_width = 1024;
            // int sampling_height = 1024;

            uint32_t n_coords = sampling_width * sampling_height;
            uint32_t n_coords_padded = (n_coords + 255) / 256 * 256;

            tcnn::GPUMemory<float> sampled_image(n_coords * 3);
            tcnn::GPUMemory<float> xs_and_ys(n_coords_padded * 2);

            std::vector<float> host_xs_and_ys(n_coords * 2);
            for (int y = 0; y < sampling_height; ++y) {
                for (int x = 0; x < sampling_width; ++x) {
                    int idx = (y * sampling_width + x) * 2;
                    host_xs_and_ys[idx + 0] = (float)(x + 0.5) / (float)sampling_width;
                    host_xs_and_ys[idx + 1] = (float)(y + 0.5) / (float)sampling_height;
                }
            }

            xs_and_ys.copy_from_host(host_xs_and_ys.data());



        }
        float tmp_loss = 0;
        uint32_t tmp_loss_counter = 0;

        std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

        // TODO3: Check training cycles, fix and find errors (taking into accout example from tiny cuda nn)
        constexpr uint32_t NumEpochs = 1;
        for (uint32_t i = 0; i < NumEpochs; ++i) {
            bool print_loss = i % 1000 == 0;
            bool visualize_learned_func = 0 < 5 && i % 1000 == 0;

            // Compute reference values at random coordinates
            {
                CURAND_CHECK_THROW(curandGenerateUniform(rng, training_batch.data(), mSharedParams.BatchSize * NumInputFeatures));
                tcnn::linear_kernel(eval_image<n_output_dims>, 0, training_stream, mSharedParams.BatchSize, texture, training_batch.data(), training_target.data());
            }

            // Training step
            float loss_value;
            {
                trainer->training_step(training_stream, training_batch, training_target, &loss_value);
            }
            tmp_loss += loss_value;
            ++tmp_loss_counter;
        }
        std::cout << tmp_loss / tmp_loss_counter << std::endl;

       
        // TODO4: The same thing - retranslate Falcor buffers (kPredictInput) to cuda Matrix and inference neural network with outputing data to some Falcor buffer
        {
            
            /*if(need to init)
                mPredictOutput = Buffer::createStructured(NumOutputFeatures, PredictXBuffer ....)*/

        }
        //save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, argv[4]);

        // TODO5: inference
        network->inference(inference_stream, inference_batch, prediction);

        
        return true;
    }

    void NeuralNetwork::setStaticParams(Program* pProgram) const
    {
        // Set compile-time constants on the given program.
        // TODO: It's unnecessary to set these every frame. It should be done lazily, but the book-keeping is complicated.
        Program::DefineList defines;
        pProgram->addDefines(defines);
    }
}
