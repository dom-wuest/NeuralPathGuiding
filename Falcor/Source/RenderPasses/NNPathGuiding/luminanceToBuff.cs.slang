struct TrainingSamples {
    float4 incidentL;
    float pdf;
};

StructuredBuffer<TrainingSamples> gTrainYBuffer;
RWStructuredBuffer<float> gTrainLuminanceBuffer;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint id = dispatchThreadID.x;

    float4 sample = gTrainYBuffer[id].incidentL;
    gTrainLuminanceBuffer[id] = sample.w / gTrainYBuffer[id].pdf;
}
