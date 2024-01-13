#include "Utils/Math/MathConstants.slangh"
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;
struct VMF
{
    float3 mu;
    float kappa;
    float A;
}

#ifndef VMF_COUNT
#define VMF_COUNT 1
#endif

struct GuidingDistribution{
    VMF lobes[VMF_COUNT];
}

float2 sample_circle(float u)
{
    float2 p;
    float phi = M_2PI * u;
    p.x = cos(phi);
    p.y = sin(phi);
    return p;
}



float4 packVMFLobe(VMF d){
    float4 packedDistribution = 0;
    packedDistribution.xy = world_to_latlong_map(normalize(d.mu));
    packedDistribution.z = d.kappa;
    packedDistribution.w = d.A;
    return packedDistribution;
}

VMF unpackVMFLobe(float4 packedDistribution){
    VMF d;
    d.kappa = packedDistribution.z;
    d.A = packedDistribution.w;
    d.mu = latlong_map_to_world(packedDistribution.xy);
    return d;
}

float3 sample_GuidingDistribution(float3 u, GuidingDistribution d, out float pdf){
    float totalA = 0.0;
    [unroll]
    for(uint i=0; i < VMF_COUNT; i++){
        totalA += d.lobes[i].A;
    }

    uint targetVMF=0;
    float acc = d.lobes[targetVMF].A;
    while(u.x*totalA >= acc){
        targetVMF+=1;
        acc += d.lobes[targetVMF].A;
    }

    pdf = d.lobes[targetVMF].A/totalA;
    float lobepdf;
    float3 res = sample_vMF(u.yz, d.lobes[targetVMF], lobepdf);
    pdf *= lobepdf;
    return res;
}

float pdf_VMF(VMF vmf, float3 x)
{
	if (vmf.kappa == 0.0){
		return .25 *M_1_PI;
    }
	else
    {
        float pdf_constant = vmf.kappa / ((2 * M_PI) * (1. - exp(-2. * vmf.kappa)));
		return pdf_constant * exp(vmf.kappa * (dot(vmf.mu, x) - 1.));
    }
}

float3 sample_vMF(float2 u, VMF vmf, out float pdf)
{
    float tmp = 1.0f - (((u.x - 1.0f) / u.x) * exp(-2.0f * vmf.kappa));
    float w = 1 + (log(u.x) + log(tmp)) / vmf.kappa;
    float w2 =  w * w;

    float2 v = sample_circle(u.y);
    v.x *= sqrt(1 - w2);
    v.y *= sqrt(1 - w2);

    
    float angle_sin = sqrt((1.0-vmf.mu.z*vmf.mu.z)+0.000001);
/*
U = axis
return numpy.array([[   0., -U[2],  U[1]],
                      [ U[2],    0., -U[0]],
	                    [-U[1],  U[0],    0.]])
*/

    float3x3 iden = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};
    float3 axis = float3(vmf.mu.y, -vmf.mu.x, 0.0);
    axis = normalize(axis);
    float3x3 m0 = vmf.mu.z*iden;
    float3x3 cross_product ={0.0, -axis.z, axis.y,
                            axis.z, 0.0, -axis.x,
                            -axis.y, axis.x, 0.0};
    float3x3 m1 = angle_sin*cross_product;
    float3x3 outer = {
            axis.x*axis.x, axis.x*axis.y, axis.x*axis.z,
            axis.y*axis.x, axis.y*axis.y, axis.y*axis.z,
            axis.z*axis.x, axis.z*axis.y, axis.z*axis.z
    };
    float3x3 m2 = (1.0-vmf.mu.z)*outer;

    // float3x3 rot_matrix = float3x3(vmf.mu.z + (1 - vmf.mu.z) * vmf.mu.y * vmf.mu.y, (1 - vmf.mu.z) * vmf.mu.x * vmf.mu.y, -angle_sin * vmf.mu.x,
    //                                (1 - vmf.mu.z) * vmf.mu.x * vmf.mu.y, vmf.mu.z + (1 - vmf.mu.z) * vmf.mu.x * vmf.mu.x, angle_sin * vmf.mu.y,
    //                                angle_sin * vmf.mu.x, -angle_sin * vmf.mu.y, vmf.mu.z);

    float3x3 rot_matrix = m0+m1+m2;
    float3 out_vec = mul(float3(v.x, v.y, w), rot_matrix); 
    out_vec = normalize(out_vec);
    pdf = pdf_VMF(vmf, out_vec);
    return out_vec;
}
