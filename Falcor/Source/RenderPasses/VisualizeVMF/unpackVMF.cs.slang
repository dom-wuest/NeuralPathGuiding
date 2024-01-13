import vMFSample;

#ifndef VMF_COUNT
#define VMF_COUNT 1
#endif

Texture2D<float4> gGuidingLobesTextures[VMF_COUNT];
RWTexture2D<float4> gMu;
RWTexture2D<float> gKappa;
RWTexture2D<float> gA;


cbuffer GeneralData 
{
    float4x4 gViewMatrix;
    float gScaleMu;
    float gScaleKappa;
    float gScaleA;
};

[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    VMF lobe = unpackVMFLobe(gGuidingLobesTextures[0][dispatchThreadID.xy]);
    float4 mu = 0;
    mu.xyz = lobe.mu;
    float4 mu_view = mul(mu, gViewMatrix);
    //mu_view = (mu_view + 1) / 2.0;
    mu_view.w = 1.0;
    gMu[dispatchThreadID.xy] = mu_view;
    gKappa[dispatchThreadID.xy] = float(lobe.kappa) * gScaleKappa;
    //gA[dispatchThreadID.xy] = float(lobe.A) * gScaleA;
    gA[dispatchThreadID.xy] = 0.0;
    if(float(lobe.kappa) > 50) gA[dispatchThreadID.xy] = 255.0;
}
