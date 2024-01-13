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
#include "PathGuiding.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Scene/HitInfo.h"
#include "Utils/NeuralNetworks/NeuralNetworkModel.h"

#include "json/json.hpp"

#define USE_SURFACE_NORMAL 1



namespace
{
    //const std::string projectpath = "F:/TeamWork/Neural/Neural/Falcor/Source/";
    const std::string projectpath = "D:/Dominik/Studium/Thesis/students-neuralgraphics/Falcor/Source/";
    const char kShaderFile[] = "RenderPasses/NNPathGuiding/NNPathGuiding.rt.slang";
    const std::string kNetworkConfig = "RenderPasses/NNPathGuiding/network_config.json";
    const std::string kNetworkConfigLuminance = "RenderPasses/NNPathGuiding/network_config_luminance.json";

    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    // The payload for the scatter rays is 8-12B.
    // The payload for the shadow rays is 4B.
    const uint32_t kMaxPayloadSizeBytes = Falcor::HitInfo::kMaxPackedSizeInBytes;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    // Render pass output channels.
    const std::string kColorOutput = "color";
    const std::string kAlbedoOutput = "albedo";
    const std::string kTimeOutput = "time";
    const std::string kVmfOutput = "vmf";
    const std::string kLuminanceOutput = "luminance";
    const std::string kIrradianceOutput = "irradiance";

    const Falcor::ChannelList kOutputChannels =
    {
        { kColorOutput,      "gOutputColor",               "Output color (linear)", true /* optional */                              },
        { kAlbedoOutput,     "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */    },
        { kTimeOutput,       "gOutputTime",                "Per-pixel execution time", true /* optional */, Falcor::ResourceFormat::R32Uint  },
        { kVmfOutput,        "gGuidingLobesTextures",      "VMF guiding distribution", true /* optional */ },
        { kLuminanceOutput,  "gOutputLuminance",           "Indirect luminance estimate", true /* optional */ },
        { kIrradianceOutput, "gOutputIrradiance",          "Incident irradiance samples", true /* optional */ },
    };

    const char kDesc[] = "Insert pass description here";
    
    
    const std::string kConvertBufferToTexture = "RenderPasses/NNPathGuiding/buffToTex.cs.slang";
    const std::string kConvertPosToBuffer = "RenderPasses/NNPathGuiding/posToBuff.cs.slang";
    const std::string kPackGuidingTargetsToBuffer = "RenderPasses/NNPathGuiding/packGuidingTargets.cs.slang";
    const std::string kExtractLuminanceToBuffer = "RenderPasses/NNPathGuiding/luminanceToBuff.cs.slang";

}

//using namespace Falcor;   

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("NNPathGuiding", kDesc, Falcor::NNPathGuiding::create);
}
#define VMF_COUNT 1

const char* Falcor::NNPathGuiding::sDesc = "Dummy Neural Network";

Falcor::NNPathGuiding::NNPathGuiding(const Dictionary& dict)
    : PathTracer(dict, kOutputChannels)
{
    mSharedParams.maxBounces = 0;
    mSharedParams.maxNonSpecularBounces = 0;

    model_guiding = new Falcor::SimpleNNModel("vMF Path Guiding", projectpath+kNetworkConfig);
    model_luminance = new Falcor::SimpleNNModel("Luminance Cache", projectpath + kNetworkConfigLuminance);
    mpConvertNNOutputPass = ComputePass::create(projectpath+kConvertBufferToTexture);
    mpPreparePredictDataPass = ComputePass::create(projectpath+kConvertPosToBuffer);
    mpPackGuidingTargetsPass = ComputePass::create(projectpath+kPackGuidingTargetsToBuffer);
    mpExtractLuminancePass = ComputePass::create(projectpath+kExtractLuminanceToBuffer);
}

Falcor::NNPathGuiding::SharedPtr Falcor::NNPathGuiding::create(Falcor::RenderContext* pRenderContext, const Falcor::Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new Falcor::NNPathGuiding(dict));
    return pPass;
}




void Falcor::NNPathGuiding::setScene(Falcor::RenderContext* pRenderContext, const Falcor::Scene::SharedPtr& pScene)
{
    PathTracer::setScene(pRenderContext, pScene);

    mpScene = pScene;
    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Procedural))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderLibrary(projectpath+kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());
        desc.addDefine("MAX_BOUNCES", std::to_string(mSharedParams.maxBounces));
        desc.addDefine("SAMPLES_PER_PIXEL", std::to_string(mSharedParams.samplesPerPixel));

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(kRayTypeScatter, desc.addMiss("scatterMiss"));
        sbt->setMiss(kRayTypeShadow, desc.addMiss("shadowMiss"));
        sbt->setHitGroupByType(kRayTypeScatter, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroupByType(kRayTypeShadow, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("", "shadowAnyHit"));

        mTracer.pProgram = RtProgram::create(desc);
    }
}

void Falcor::NNPathGuiding::execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData)
{

    if (!mpScene)
    {
        return;
    }

    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;
    
    if (!model_guiding)
    {
        std::cout << "ERROR!!" << std::endl;
    }
    if (!mpCounterBuffer) {
        mpCounterBuffer = Buffer::create(1 * sizeof(uint32_t), ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!mpFenceDataForPrediction) mpFenceDataForPrediction = GpuFence::create(true);
    if (!mpFencePredictionResults) mpFencePredictionResults = GpuFence::create(true);
    if (!mpFencePredictionLuminanceOutput) mpFencePredictionLuminanceOutput = GpuFence::create(true);
    if (!mpFenceDataForTrainingVMF) mpFenceDataForTrainingVMF = GpuFence::create(true);
    if (!mpFenceDataForTrainingLuminance) mpFenceDataForTrainingLuminance = GpuFence::create(true);
    if (!mpFencePredictionLuminance) mpFencePredictionLuminance = GpuFence::create(true);
    if (!mpFenceExtractLuminance) mpFenceExtractLuminance = GpuFence::create(true);
    if (!mpFencePackLuminance) mpFencePackLuminance = GpuFence::create(true);


    auto t = renderData["posW"]->asTexture();
    mScreenDim.x = t->getWidth(0);
    mScreenDim.y = t->getHeight(0);
    
    
    
    
    uint32_t w = t->getWidth();
    uint32_t h = t->getHeight();
    uint32_t nElements = w * h;
    uint32_t maxTrainingElements = nElements;
    if (predXBuffer == nullptr) {
#if (USE_SURFACE_NORMAL==1)
        predXBuffer = Falcor::Buffer::createStructured(7 * sizeof(float), nElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
#else
        predXBuffer = Falcor::Buffer::createStructured(4 * sizeof(float), nElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
#endif
        predLuminanceBuffer = Falcor::Buffer::createStructured(sizeof(float), nElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        predVMFBuffer = Falcor::Buffer::createStructured(VMF_COUNT * 4 * sizeof(float), nElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        trainVMFBuffer = Falcor::Buffer::createStructured(6*sizeof(float), maxTrainingElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        trainLuminanceBuffer = Falcor::Buffer::createStructured(sizeof(float), maxTrainingElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        //guidingLobesTextures = Falcor::Texture::create2D(w, h, ResourceFormat::RGBA32Float, VMF_COUNT, 1, nullptr, Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource);
        guidingLobesTextures = renderData["vmf"]->asTexture();
        luminanceOutputTexture = renderData["luminance"]->asTexture();
        irradianceTexture = renderData["irradiance"]->asTexture();
        luminanceOutputBuffer = Falcor::Buffer::createStructured(sizeof(float), nElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        trainSamplesBuffer = Falcor::Buffer::createStructured(5 * sizeof(float), maxTrainingElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
    }

    mpPreparePredictDataPass["gPosW"] = renderData["posW"]->asTexture();
    mpPreparePredictDataPass["gNormalW"] = renderData["normalW"]->asTexture();
    mpPreparePredictDataPass["gPredXBuffer"] = predXBuffer;
    mpPreparePredictDataPass["GeneralData"]["bbStart"] = mpScene->getSceneBounds().minPoint;
    mpPreparePredictDataPass["GeneralData"]["bbEnd"] = mpScene->getSceneBounds().maxPoint;
    {
        PROFILE("mpPreparePredictDataPass::execute()");
        mpPreparePredictDataPass->execute(pRenderContext, Falcor::uint3(w, h, uint32_t(1)));
    }


    // Wait getting data for predicting in cuda
    pRenderContext->flush(false);
    uint64_t signalValue = mpFenceDataForPrediction->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

    {
        PROFILE("model_guiding::predict()");
        model_guiding->predict(predXBuffer, predVMFBuffer, nElements, mpFenceDataForPrediction, mpFencePredictionResults, signalValue);
        // Waiting predictions of nn in renderpass
        mpFencePredictionResults->syncGpu(pRenderContext->getLowLevelData()->getCommandQueue());
    }

    if (bPredictLuminancePerPixel){
        // Wait getting data for predicting in cuda
        //pRenderContext->flush(false);
        //signalValue = mpFenceDataForPrediction->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        model_luminance->predict(predXBuffer, luminanceOutputBuffer, nElements, mpFenceDataForPrediction, mpFencePredictionLuminanceOutput, signalValue);
        mpFencePredictionLuminanceOutput->syncGpu(pRenderContext->getLowLevelData()->getCommandQueue());
    }
    
    mpConvertNNOutputPass["gPredVMFs"] = predVMFBuffer;
    mpConvertNNOutputPass["gPredLuminance"] = luminanceOutputBuffer;
    mpConvertNNOutputPass["gGuidingLobesTextures"] = guidingLobesTextures;
    mpConvertNNOutputPass["gOutputLuminance"] = luminanceOutputTexture;
    {
        PROFILE("mpConvertNNOutputPass::execute()");
        mpConvertNNOutputPass->execute(pRenderContext, Falcor::uint3(w, h, uint32_t(1)));
    }
    


    // Set compile-time constants.
    RtProgram::SharedPtr pProgram = mTracer.pProgram;
    setStaticParams(pProgram.get());

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));
    pProgram->addDefines(getValidResourceDefines(mOutputChannels, renderData));

    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (pProgram->addDefines(mpEmissiveSampler->getDefines())) mTracer.pVars = nullptr;
    }

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set shared data into parameter block.
    setTracerData(renderData);

    mTracer.pVars["gTrainY"] = trainSamplesBuffer;
    mTracer.pVars["gTrainX"] = predXBuffer;
    mTracer.pVars["gGuidingLobesTextures"] = guidingLobesTextures;
    mTracer.pVars["gOutputLuminance"] = luminanceOutputTexture;
    mTracer.pVars["gOutputIrradiance"] = irradianceTexture;
    mTracer.pVars["GeneralData"]["bbStart"] = mpScene->getSceneBounds().minPoint;
    mTracer.pVars["GeneralData"]["bbEnd"] = mpScene->getSceneBounds().maxPoint;
    mTracer.pVars["GeneralData"]["guiding"] = bGuiding;
    mTracer.pVars["GeneralData"]["explorationRatio"] = bExplorationRatio;
    mTracer.pVars["GeneralData"]["indirectOnly"] = bIndirectOnly;
    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto var = mTracer.pVars->getRootVar();
            var[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : mInputChannels) bind(channel);
    for (auto channel : mOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);

    mpPixelDebug->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    mpPixelStats->prepareProgram(pProgram, mTracer.pVars->getRootVar());

    // Spawn the rays.
    {
        PROFILE("MegakernelPathTracer::execute()_RayTrace");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }

    // Get number of added training samples
    pRenderContext->copyBufferRegion(mpCounterBuffer.get(), 0, trainSamplesBuffer->getUAVCounter().get(), 0, 4);
    uint32_t* uavCounters = (uint32_t*)mpCounterBuffer->map(Buffer::MapType::Read);
    uint32_t NumTrainingElements = uavCounters[0];
    mpCounterBuffer->unmap();

    
    
    if (model_guiding->getTrainFlag() && NumTrainingElements > 0) {
        // Wait getting data for training in cuda
        pRenderContext->flush(false);
        signalValue = mpFenceDataForTrainingVMF->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        uint32_t realNumTrainElem = (NumTrainingElements / 128) * 128;
        {
            PROFILE("model_luminance::predict()");
            model_luminance->predict(predXBuffer, predLuminanceBuffer, realNumTrainElem, mpFenceDataForTrainingVMF, mpFencePredictionLuminance, signalValue);
        }
        // Waiting predictions of luminance nn in shader
        mpFencePredictionLuminance->syncGpu(pRenderContext->getLowLevelData()->getCommandQueue());

        // TODO: pack path data and predicted luminance for training of model_guiding: (w, L, A), then fit

        mpPackGuidingTargetsPass["gLuminance"] = predLuminanceBuffer;
        mpPackGuidingTargetsPass["gSamples"] = trainSamplesBuffer;
        mpPackGuidingTargetsPass["gTargets"] = trainVMFBuffer;
        {
            PROFILE("mpPackGuidingTargetPass::execute()");
            mpPackGuidingTargetsPass->execute(pRenderContext, Falcor::uint3(realNumTrainElem, uint32_t(1), uint32_t(1)));
        }

        pRenderContext->flush(false);
        signalValue = mpFencePackLuminance->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        {
            PROFILE("model_guiding::fit()");
            model_guiding->fit(predXBuffer, trainVMFBuffer, realNumTrainElem, mpFencePackLuminance, signalValue, 4, 6);
        }
    }

    if (model_luminance->getTrainFlag() && NumTrainingElements > 0) {
        pRenderContext->flush(false);
        signalValue = mpFenceDataForTrainingLuminance->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        mpExtractLuminancePass["gTrainYBuffer"] = trainSamplesBuffer;
        mpExtractLuminancePass["gTrainLuminanceBuffer"] = trainLuminanceBuffer;
        {
            PROFILE("mpExtractLuminancePass::execute()");
            mpExtractLuminancePass->execute(pRenderContext, Falcor::uint3(NumTrainingElements, uint32_t(1), uint32_t(1)));
        }
        pRenderContext->flush(false);
        signalValue = mpFenceExtractLuminance->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        {
            PROFILE("model_luminance::fit()");
            model_luminance->fit(predXBuffer, trainLuminanceBuffer, NumTrainingElements, mpFenceExtractLuminance, signalValue, 4);
        }
    }
    pRenderContext->clearUAVCounter(trainSamplesBuffer->asBuffer(), 0);

    // Call shared post-render code.
    endFrame(pRenderContext, renderData);
}

void Falcor::NNPathGuiding::renderUI(Gui::Widgets& widget)
{
    Falcor::PathTracer::renderUI(widget);
    
    if (auto lcGroup = widget.group("Luminance-Cache", true))
    {
        model_luminance->renderUI(widget);
        lcGroup.checkbox("Output predicted Luminance per Pixel", bPredictLuminancePerPixel);
    }
    if (auto nnGroup = widget.group("Path-Guiding", true))
    {
        model_guiding->renderUI(widget);
        nnGroup.checkbox("Path-Guiding", bGuiding);
        nnGroup.slider("Exploration-Ratio", bExplorationRatio,0.0f,1.0f);
        nnGroup.tooltip("Exploration-Ratio=1.0 will distribute all new training samples according to the "
            "predicted vMF to explore incident light we discovered previously."
            "Exploration-Ratio=0.0 will not use vMF at all and distribute training samples uniformly to discover "
            "previosly unkown incident light."
        );
        nnGroup.checkbox("Learn indirect light only", bIndirectOnly);
    }
}

void Falcor::NNPathGuiding::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(var);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    ProgramReflection::SharedConstPtr pReflection = mTracer.pProgram->getReflector();
    ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = ParameterBlock::create(pBlockReflection);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(mTracer.pParameterBlock["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mTracer.pVars->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);
}

void Falcor::NNPathGuiding::setTracerData(const RenderData& renderData)
{
    auto pBlock = mTracer.pParameterBlock;
    assert(pBlock);

    // Upload parameters struct.
    pBlock["params"].setBlob(mSharedParams);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = mpEmissiveSampler->setShaderData(pBlock["emissiveSampler"]);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}
