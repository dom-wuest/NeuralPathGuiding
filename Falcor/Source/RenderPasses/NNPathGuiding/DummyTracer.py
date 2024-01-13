def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("NNPathGuiding.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    NNPathGuiding = createPass("NNPathGuiding", {
        'params': PathTracerParams(useVBuffer=0, useNEE=0)
        })
    g.addPass(NNPathGuiding, "NNPathGuiding")
    g.addEdge("GBufferRT.vbuffer", "NNPathGuiding.vbuffer")      # Required by ray footprint.
    g.addEdge("GBufferRT.posW", "NNPathGuiding.posW")
    g.addEdge("GBufferRT.normW", "NNPathGuiding.normalW")
    g.addEdge("GBufferRT.tangentW", "NNPathGuiding.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "NNPathGuiding.faceNormalW")
    g.addEdge("GBufferRT.viewW", "NNPathGuiding.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "NNPathGuiding.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "NNPathGuiding.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "NNPathGuiding.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "NNPathGuiding.mtlParams")
    g.addEdge("NNPathGuiding.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None
