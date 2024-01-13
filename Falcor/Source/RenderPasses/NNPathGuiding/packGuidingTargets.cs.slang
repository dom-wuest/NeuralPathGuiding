StructuredBuffer<float> gLuminance;

struct TrainingSamples {
    float4 incidentL;
    float pdf;
};

StructuredBuffer<TrainingSamples> gSamples;



struct TargetVectorNN {
    float3 dir;
    float L;
    float A;
    float pdf;
};

RWStructuredBuffer<TargetVectorNN> gTargets;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint id = dispatchThreadID.x;

    float4 sample = gSamples[id].incidentL;
    float A = gLuminance[id];
    float pdf = gSamples[id].pdf;
    gTargets[id] = { sample.xyz, sample.w, A, pdf };
}
