
#pragma once
#include "FalcorCUDA.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"

#include "Falcor.h"

namespace Falcor {
    class SimpleNNModel;

    class NNPathGuiding : public PathTracer
    {
    public:
        using SharedPtr = std::shared_ptr<NNPathGuiding>;

        /** Create a new render pass object.
            \param[in] pRenderContext The render context.
            \param[in] dict Dictionary of serialized parameters.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(Falcor::RenderContext* pRenderContext = nullptr, const Falcor::Dictionary& dict = {});

        virtual std::string getDesc() override { return sDesc; }
    
        virtual void execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) override;
        virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
        virtual void renderUI(Gui::Widgets& widget) override;
    
        static const char* sDesc;

    private:
        NNPathGuiding(const Dictionary& dict);
        Falcor::Buffer::SharedPtr predXBuffer;
        Falcor::Buffer::SharedPtr predLuminanceBuffer;
        Falcor::Buffer::SharedPtr predVMFBuffer;
        Falcor::Buffer::SharedPtr trainVMFBuffer;
        Falcor::Buffer::SharedPtr trainLuminanceBuffer;
        Falcor::Texture::SharedPtr guidingLobesTextures;
        Falcor::Texture::SharedPtr luminanceOutputTexture;
        Falcor::Texture::SharedPtr irradianceTexture;
        Falcor::Buffer::SharedPtr luminanceOutputBuffer;
        Falcor::Buffer::SharedPtr trainSamplesBuffer;


        
        Falcor::ComputePass::SharedPtr          mpGenerateTrainingDataPass;
        Falcor::ComputePass::SharedPtr          mpConvertNNOutputPass;
        Falcor::ComputePass::SharedPtr          mpPreparePredictDataPass;
        Falcor::ComputePass::SharedPtr          mpPackGuidingTargetsPass;
        Falcor::ComputePass::SharedPtr          mpExtractLuminancePass;

        Falcor::SimpleNNModel* model_guiding;
        Falcor::SimpleNNModel* model_luminance;

        Falcor::Buffer::SharedPtr mpCounterBuffer;
        uint2 mScreenDim = { 1920, 1080 };
        Falcor::Scene::SharedPtr mpScene;

        GpuFence::SharedPtr mpFenceDataForPrediction;
        GpuFence::SharedPtr mpFencePredictionResults;
        GpuFence::SharedPtr mpFencePredictionLuminanceOutput;
        GpuFence::SharedPtr mpFenceDataForTrainingVMF;
        GpuFence::SharedPtr mpFenceDataForTrainingLuminance;
        GpuFence::SharedPtr mpFenceExtractLuminance;
        GpuFence::SharedPtr mpFencePackLuminance;
        GpuFence::SharedPtr mpFencePredictionLuminance;

        bool bGuiding= true;
        bool bIndirectOnly = false;
        bool bPredictLuminancePerPixel = false;
        float bExplorationRatio = 0.5;

        void recreateVars() override { mTracer.pVars = nullptr; }
        void prepareVars();
        void setTracerData(const RenderData& renderData);

        // Ray tracing program.
        struct
        {
            RtProgram::SharedPtr pProgram;
            RtBindingTable::SharedPtr pBindingTable;
            RtProgramVars::SharedPtr pVars;
            ParameterBlock::SharedPtr pParameterBlock;      ///< ParameterBlock for all data.
        } mTracer;
    };
}
