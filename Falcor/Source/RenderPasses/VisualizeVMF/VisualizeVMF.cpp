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
#include "VisualizeVMF.h"


namespace
{
    const char kDesc[] = "Insert pass description here";

    const std::string kUnpackCompute = "RenderPasses/VisualizeVMF/unpackVMF.cs.slang";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("VisualizeVMF", kDesc, VisualizeVMF::create);
}

VisualizeVMF::VisualizeVMF() : RenderPass() {
    mpUnpackCP = Falcor::ComputePass::create(kUnpackCompute);
}

VisualizeVMF::SharedPtr VisualizeVMF::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new VisualizeVMF);
    return pPass;
}

std::string VisualizeVMF::getDesc() { return kDesc; }

Dictionary VisualizeVMF::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection VisualizeVMF::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    auto& outputMu = reflector.addOutput("mu", "mean direction of vMF");
    outputMu.bindFlags(Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
    outputMu.format(Falcor::ResourceFormat::RGBA32Float);
    auto& outputKappa = reflector.addOutput("kappa", "concentration of vMF");
    outputKappa.bindFlags(Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
    outputKappa.format(Falcor::ResourceFormat::R32Float);
    auto& outputA = reflector.addOutput("a", "normalization of vMF");
    outputA.bindFlags(Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess);
    outputA.format(Falcor::ResourceFormat::R32Float);
    reflector.addInput("vmf", "vMF lobes in packed format");
    return reflector;
}

void VisualizeVMF::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) {
    mpScene = pScene;
}

void VisualizeVMF::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto &view_matrix = mpScene->getCamera()->getViewMatrix();


    // renderData holds the requested resources
    const auto& pVMF = renderData["vmf"]->asTexture();
    uint32_t w = pVMF->getWidth();
    uint32_t h = pVMF->getHeight();

    

    mpUnpackCP["gGuidingLobesTextures"] = pVMF;
    mpUnpackCP["gMu"] = renderData["mu"]->asTexture();
    mpUnpackCP["gKappa"] = renderData["kappa"]->asTexture();
    mpUnpackCP["gA"] = renderData["a"]->asTexture();
    mpUnpackCP["GeneralData"]["gViewMatrix"] = view_matrix;
    mpUnpackCP["GeneralData"]["gScaleMu"] = scaleMu;
    mpUnpackCP["GeneralData"]["gScaleKappa"] = scaleKappa;
    mpUnpackCP["GeneralData"]["gScaleA"] = scaleA;
    mpUnpackCP->execute(pRenderContext, Falcor::uint3(w, h, uint32_t(1)));

    
    
}

void VisualizeVMF::renderUI(Gui::Widgets& widget)
{
    widget.slider("Scale Mu", scaleMu, 0.0001f, 1.0f);
    widget.slider("Scale Kappa", scaleKappa, 0.0001f, 1.0f);
    widget.slider("Scale A", scaleA, 0.0001f, 1.0f);
}
