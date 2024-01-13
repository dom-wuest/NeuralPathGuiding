/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   identity.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the identity encoding (output == input).
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void identity(
	const uint32_t num_elements,
	const uint32_t num_to_encode,
	const uint32_t num_to_pad,
	const float scale,
	const float offset,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out,
	float* __restrict__ dy_dx,
	const int* __restrict__ mapping = nullptr,
	float* bbStart = nullptr,
	float* bbEnd = nullptr
	)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t fan_out = num_to_encode + num_to_pad;
	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out ;
	if (j >= num_to_encode) {
		data_out(i)[j] = 1;
	} else {
#if 1
		float x = (mapping == nullptr ) ? data_in(i)[j] : data_in(mapping[i])[j];
		 /*if(bbStart != nullptr && j < 3){
		 	x = clamp(x, bbStart[j], bbEnd[j]);
		 	x = (x-bbStart[j])/(bbEnd[j]-bbStart[j])*2-1.0;
		 }*/
#else
		const float x = data_in(i)[j];
#endif
		//const float x = (mapping == nullptr) ? data_in.ptr[i*14+j] : data_in.ptr[mapping[i] * 14+j];
		data_out(i)[j] = x*scale + offset;
		if (dy_dx != nullptr) {
			dy_dx[i * num_to_encode + j] = scale;
		}
	}
}

template <typename T>
__global__ void identity_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const float scale,
	PitchedPtr<const T> dL_dy,
	const float* dy_dx,
	PitchedPtr<float> dL_dx)
{
	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / n_dims_to_encode;
	const uint32_t j = output_index - i * n_dims_to_encode;

	// The identity encoding can simply pass through the derivative.
	dL_dx(i)[j] = (T)((float)dL_dy(i)[j] * scale);
}

template <typename T>
class IdentityEncoding : public Encoding<T> {
public:
	IdentityEncoding(uint32_t n_dims_to_encode, float scale = 1.0f, float offset = 0.0f)
	: m_n_dims_to_encode{n_dims_to_encode}, m_scale{scale}, m_offset{offset} {
		m_n_padded_output_dims = m_n_output_dims = m_n_dims_to_encode;
	}

	void encode(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const float> inputs,
		PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false,
		int* mapping = nullptr,
		float* bbStart = nullptr,
		float* bbEnd = nullptr
	) const override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		linear_kernel(identity<T>, 0, stream,
			num_elements * num_encoded_dims(),
			m_n_dims_to_encode,
			m_n_to_pad,
			m_scale,
			m_offset,
			inputs,
			outputs,
			dy_dx,
			mapping, bbStart, bbEnd
		);
	}


	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x num_elements
		PitchedPtr<float> dL_dx, // Same shape as inputs
		PitchedPtr<const float> inputs,
		bool accumulate_param_gradients
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		// Can't compute input gradients if insufficient info is available
		if (!dy_dx || !dL_dx) {
			return;
		}

		linear_kernel(identity_backward<T>, 0, stream,
			num_elements * m_n_dims_to_encode,
			m_n_dims_to_encode,
			m_scale,
			dL_dy,
			dy_dx,
			dL_dx
		);
	}

	uint32_t num_dims_to_encode() const override {
		return m_n_dims_to_encode;
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return m_n_dims_to_encode;
	}

	void set_alignment(uint32_t alignment) override {
		alignment = lcm(alignment, min_alignment());
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
	}

	uint32_t min_alignment() const override {
		return 1;
	}

private:
	uint32_t m_n_dims_to_encode;

	float m_scale;
	float m_offset;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
