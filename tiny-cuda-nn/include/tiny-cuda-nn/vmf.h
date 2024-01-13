#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/object.h>

#define VMF_KAPPA_MINV 0.1
#define VMF_A_MINV 0.001
#define VMF_COUNT 1


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void apply_vmf(const uint32_t num_elements, const uint32_t stride, const uint32_t dims, T* __restrict__ data_in)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	uint32_t idx = i % dims;
	uint32_t elem = i / dims;

	float x = (float)data_in[elem * stride + idx];
	if (idx >= 4)	
		return;

	if (idx < 3) {
		x = 2*__frcp_rn(1.0f + __expf(-x))-1.0;
	}
	else {
		//x = x * x;
		//if (idx == 3)
		//	x += VMF_KAPPA_MINV;
		x = __expf(x);
		//if (idx == 4)
		//	x += VMF_A_MINV;
	}
	data_in[elem * stride + idx] = (T)x;
}

TCNN_NAMESPACE_END
