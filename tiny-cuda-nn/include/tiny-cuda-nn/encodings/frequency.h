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

/** @file   frequency.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the frequency encoding of NeRF [Mildenhall et al. 2020].
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
__global__ void frequency_encoding(
	const uint32_t num_elements,
	const uint32_t n_frequencies,
	const uint32_t num_to_encode,
	const uint32_t num_to_pad,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out,
	float* __restrict__ dy_dx,
	int* mapping = nullptr,
	float* bbStart = nullptr,
	float* bbEnd = nullptr,
	const uint32_t freqShift = 0,
	const bool firstIdentity=false,
	const int32_t hashNormalization=-1,
	const float t=-1.0)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t fan_out_encoded = num_to_encode * n_frequencies;
	const uint32_t fan_out = fan_out_encoded + num_to_pad;

	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out;

	/* Layout of outputs (for each input record):
	 *     frequency-encoded input dimension 0
	 *     frequency-encoded input dimension 1
	 *     frequency-encoded input dimension ...
	 *     passthrough inputs
	 *     padding (value 1.f)
	 */
	if (j >= fan_out_encoded) {
		data_out(i)[j] = 1;
	} else {
		/* Layout of encoded features (e.g. when inputs abcd.. are XYZ positions):
		 *     sin(a.x), cos(a.x) sin(2pi a.x), cos(2pi a.x) sin(4pi a.x) ...
		 *     sin(a.y), cos(a.y) sin(2pi a.y), cos(2pi a.y) sin(4pi a.y) ...
		 *     sin(a.z), cos(a.z) sin(2pi a.z), cos(2pi a.z) sin(4pi a.z) ...
		 *     (passthrough features)
		 *     (padding)
		 */
		const uint32_t encoded_input_feature_i = j / (n_frequencies);
		 uint32_t log2_frequency = (j-uint32_t(firstIdentity)) % n_frequencies+freqShift;

		const float phase_shift = 0;

		
		
		#if 1
			float x = (mapping == nullptr) ? data_in(i)[encoded_input_feature_i] : data_in(mapping[i])[encoded_input_feature_i];
			if(hashNormalization != -1){
				if(bbStart != nullptr && 1){
					const float xMin = bbStart[encoded_input_feature_i];
					const float xMax = bbEnd[encoded_input_feature_i];
					x = clamp(x, xMin, xMax);
					x = (x-xMin)/(xMax-xMin);
					x = x*2-1.0;
				}
			}
			else{
				if(firstIdentity && j % n_frequencies != 0){
					#if 1
						float cellSize = hashNormalization;
						float t = x*cellSize;
						x = t-uint32_t(t);
						x = x*2-1.0;
					#else
						 
					#endif
				}
			}
		#else
			float x = data_in(i)[encoded_input_feature_i];
		#endif

		if(firstIdentity && j % n_frequencies == 0){
			#if 1
				
				//x = uint32_t(t);
			#endif

			data_out(i)[j] = x*2-1.0;
			if (dy_dx != nullptr) {
				dy_dx[i * fan_out_encoded + j] = 1.0;
			}
		}
		else{
			



			float fac = 1.0f;
			#if 0
			if (t == -1.0f)
				fac = 1.0f;
			else {
				uint32_t cur_freq = log2_frequency - freqShift;
				float inv_n_freqs = 1.0f / n_frequencies;
				fac = (cur_freq == 0) ? 1.0f : saturate((t - cur_freq* inv_n_freqs)* inv_n_freqs);
				log2_frequency = (log2_frequency == 0) ? log2_frequency : (log2_frequency-1)*(1.0-fac)+fac*log2_frequency;
			}
			#endif

			x = scalbnf(x, log2_frequency);
			const float input = x * PI + phase_shift;
			float r = (T)__sinf(input);

			data_out(i)[j] = r*fac;

			if (dy_dx != nullptr) {
				dy_dx[i * fan_out_encoded + j] = scalbnf(1.0f, log2_frequency) * PI * __cosf(input);
			}
		}
	}
}

template <typename T>
__global__ void frequency_encoding_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_frequencies,
	PitchedPtr<const T> dL_dy,
	const float* dy_dx,
	PitchedPtr<float> dL_dx
) {
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / n_dims_to_encode;
	const uint32_t j = encoded_index - i * n_dims_to_encode;

	const uint32_t outputs_per_input = n_frequencies;

	float result = 0;
	for (int k = 0; k < outputs_per_input; ++k) {
		result += (float)dL_dy(i)[j * outputs_per_input + k] * dy_dx[i * n_dims_to_encode * outputs_per_input + j * outputs_per_input + k];
	}
	dL_dx(i)[j] = result;
}

template <typename T>
class FrequencyEncoding : public Encoding<T> {
public:
	FrequencyEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode, uint32_t freqShift = 0, bool firstIdentity=false, int32_t hashNormalization=-1, int32_t freqFrameDelay=-1.0)
	: m_n_frequencies{n_frequencies}, m_n_dims_to_encode{n_dims_to_encode}, freqShift{freqShift}, firstIdentity{firstIdentity}, hashNormalization{hashNormalization}, freqFrameDelay(freqFrameDelay)
	{
		m_n_padded_output_dims = m_n_output_dims = m_n_dims_to_encode * m_n_frequencies;
		frames = 0;
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
		


		float t = (freqFrameDelay == -1) ? -1 : frames / freqFrameDelay;
		if (t > 1.0f) {
			t = 1.0f;
		}

		linear_kernel(frequency_encoding<T>, 0, stream,
			num_elements * num_encoded_dims(),
			m_n_frequencies,
			m_n_dims_to_encode,
			m_n_to_pad,
			inputs,
			outputs,
			dy_dx,
			mapping, bbStart, bbEnd,
			freqShift, firstIdentity,hashNormalization,
			t
		);
		frames += 1;
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

		linear_kernel(frequency_encoding_backward<T>, 0, stream,
			num_elements * m_n_dims_to_encode,
			m_n_dims_to_encode,
			m_n_frequencies,
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
		return m_n_dims_to_encode * m_n_frequencies;
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
	uint32_t m_n_frequencies;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;
	uint32_t freqShift = 0;
	bool firstIdentity = false;
	int32_t hashNormalization = -1;
	mutable uint32_t frames = 0;
	int32_t freqFrameDelay=-1;
};

TCNN_NAMESPACE_END
