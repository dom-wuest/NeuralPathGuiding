# Graphs
from falcor import *

def render_graph_PathTracerGraph():
    g = RenderGraph('PathTracerGraph')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('VisualizeVMF.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('NNPathGuiding.dll')
    AccumulatePass = createPass('AccumulatePass', {'enabled': False, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    AccumulatePassIrr = createPass('AccumulatePass', {'enabled': False, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePassIrr, 'AccumulatePassIrr')
    VisualizeVMFPass = createPass('VisualizeVMF')
    g.addPass(VisualizeVMFPass, 'VisualizeVMFPass')
    ToneMappingPass = createPass('ToneMapper', {'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority})
    g.addPass(ToneMappingPass, 'ToneMappingPass')
    GBufferRT = createPass('GBufferRT', {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'texLOD': TexLODMode.Mip0, 'useTraceRayInline': False})
    g.addPass(GBufferRT, 'GBufferRT')
    SideBySidePass = createPass('SideBySidePass', {'splitLocation': 0.5, 'showTextLabels': False, 'leftLabel': 'Left side', 'rightLabel': 'Right side'})
    g.addPass(SideBySidePass, 'SideBySidePass')
    NNPathGuiding = createPass('NNPathGuiding', {'params': PathTracerParams(samplesPerPixel=1, lightSamplesPerVertex=1, maxBounces=0, maxNonSpecularBounces=0, useVBuffer=0, useAlphaTest=1, adjustShadingNormals=0, forceAlphaOne=1, clampSamples=0, clampThreshold=10.0, specularRoughnessThreshold=0.25, useBRDFSampling=0, useNEE=0, useMIS=1, misHeuristic=1, misPowerExponent=2.0, useRussianRoulette=0, probabilityAbsorption=0.20000000298023224, useFixedSeed=0, useNestedDielectrics=1, useLightsInDielectricVolumes=0, disableCaustics=0, rayFootprintMode=0, rayConeMode=2, rayFootprintUseRoughness=0), 'sampleGenerator': 1, 'emissiveSampler': EmissiveLightSamplerType.LightBVH, 'uniformSamplerOptions': LightBVHSamplerOptions(buildOptions=LightBVHBuilderOptions(splitHeuristicSelection=SplitHeuristic.BinnedSAOH, maxTriangleCountPerLeaf=10, binCount=16, volumeEpsilon=0.0010000000474974513, splitAlongLargest=False, useVolumeOverSA=False, useLeafCreationCost=True, createLeavesASAP=True, allowRefitting=True, usePreintegration=True, useLightingCones=True), useBoundingCone=True, useLightingCone=True, disableNodeFlux=False, useUniformTriangleSampling=True, solidAngleBoundMethod=SolidAngleBoundMethod.Sphere)})
    g.addPass(NNPathGuiding, 'NNPathGuiding')
    g.addEdge('GBufferRT.vbuffer', 'NNPathGuiding.vbuffer')
    g.addEdge('GBufferRT.posW', 'NNPathGuiding.posW')
    g.addEdge('GBufferRT.normW', 'NNPathGuiding.normalW')
    g.addEdge('GBufferRT.tangentW', 'NNPathGuiding.tangentW')
    g.addEdge('GBufferRT.faceNormalW', 'NNPathGuiding.faceNormalW')
    g.addEdge('GBufferRT.viewW', 'NNPathGuiding.viewW')
    g.addEdge('GBufferRT.diffuseOpacity', 'NNPathGuiding.mtlDiffOpacity')
    g.addEdge('GBufferRT.specRough', 'NNPathGuiding.mtlSpecRough')
    g.addEdge('GBufferRT.emissive', 'NNPathGuiding.mtlEmissive')
    g.addEdge('GBufferRT.matlExtra', 'NNPathGuiding.mtlParams')
    g.addEdge('NNPathGuiding.color', 'AccumulatePass.input')
    g.addEdge('NNPathGuiding.irradiance', 'AccumulatePassIrr.input')
    g.addEdge('AccumulatePass.output', 'ToneMappingPass.src')
    g.addEdge('AccumulatePass.output', 'SideBySidePass.leftInput')
    g.addEdge('VisualizeVMFPass.mu', 'SideBySidePass.rightInput')
    g.addEdge('NNPathGuiding.vmf', 'VisualizeVMFPass.vmf')
    g.markOutput('ToneMappingPass.dst')
    g.markOutput('AccumulatePass.output')
    g.markOutput('AccumulatePassIrr.output')
    g.markOutput('VisualizeVMFPass.mu')
    g.markOutput('VisualizeVMFPass.kappa')
    g.markOutput('NNPathGuiding.luminance')
    g.markOutput('NNPathGuiding.irradiance')
    g.markOutput('SideBySidePass.output')
    return g
m.addGraph(render_graph_PathTracerGraph())

# Scene
m.loadScene('TestScenes/../../../Scenes/CornellBoxDual/CornellBoxDual.pyscene')
m.scene.renderSettings = SceneRenderSettings(useEnvLight=True, useAnalyticLights=True, useEmissiveLights=True, useVolumes=True)
m.scene.camera.position = float3(0.000000,0.280000,1.200000)
m.scene.camera.target = float3(0.000000,0.280000,0.000000)
m.scene.camera.up = float3(0.000000,1.000000,0.000000)
m.scene.cameraSpeed = 1.0

# Window Configuration
m.resizeSwapChain(1920, 1061)
m.ui = True

# Clock Settings
m.clock.time = 0
m.clock.framerate = 0
# If framerate is not zero, you can use the frame property to set the start frame
# m.clock.frame = 0

# Frame Capture
m.frameCapture.outputDir = '.'
m.frameCapture.baseFilename = 'Mogwai'

