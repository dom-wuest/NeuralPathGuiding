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

 /** @file   relative_l2_luminance.h
  *  @author Thomas Müller, NVIDIA
  *  @brief  Hacky implementation of the relative l2 loss based on the LUMINANCE of a six-channel prediction
  */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void relative_l2_luminance_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	const float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	const float* __restrict__ data_pdf = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t intra_elem_idx = i % stride;
	const uint32_t inter_elem_idx = i / stride;
	if (intra_elem_idx >= dims) {
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;

	const uint32_t n_total = n_elements / stride * dims;

	const float prediction = (float)predictions[i];

	float r = clamp((float)predictions[i - intra_elem_idx + 0], 0.0f, 10000.0f);
	float g = clamp((float)predictions[i - intra_elem_idx + 1], 0.0f, 10000.0f);
	float b = clamp((float)predictions[i - intra_elem_idx + 2], 0.0f, 10000.0f);
	if (dims >= 6) {
		r += (float)predictions[i - intra_elem_idx + 3];
		g += (float)predictions[i - intra_elem_idx + 4];
		b += (float)predictions[i - intra_elem_idx + 5];
	}


	const float Fac = 1.0;
#define SQRT 0

	float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
#if !SQRT
	luminance = luminance;
#endif
	//const float prediction_sq_plus_epsilon =sqrt(sqrt(sqrt(sqrt(luminance+0.000001f)+0.0000001f)+ 0.0000001f ) + 0.0000001f )+ 0.001f;
	const float prediction_sq_plus_epsilon =luminance + 0.01f;
	//const float prediction_sq_plus_epsilon = 1.0f;

	const float pdf = data_pdf ? data_pdf[target_idx] : 1;

#if SQRT
	const float difference = sqrt(clamp(prediction*Fac, 0.0f, 10000.0f) + 0.000001) - sqrt(targets[target_idx]*Fac + 0.000001) / pdf;
#else
	const float difference = prediction - targets[target_idx] / pdf;
#endif

	values[i] = difference * difference / prediction_sq_plus_epsilon / n_total;

	float gradient = 2 * difference / prediction_sq_plus_epsilon;
	gradients[i] = (T)(loss_scale * gradient / n_total);
}


template <typename T>
__global__ void relative_l2_luminance_loss_lod(
	const uint32_t n_elements_padded,
	const uint32_t n_elements,
	const int* __restrict__ mapping,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	 float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	 float* __restrict__ data_pdf = nullptr,
	uint32_t LOD = 0,
	float ema = 1.0f
) {

	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t realI = i;
	if (i >= n_elements_padded) return;
	if (i >= n_elements) {
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

	
	const uint32_t intra_elem_idx = i % stride;
	const uint32_t inter_elem_idx = i / stride;
	if (intra_elem_idx >= dims) {
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

#if 1
	// Maybe it will work better:
	uint32_t target_idx = inter_elem_idx;
if(mapping != nullptr)
	target_idx = mapping[target_idx]*dims+intra_elem_idx;
else
	target_idx = inter_elem_idx * dims + intra_elem_idx;

#else
	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;
#endif

	const uint32_t n_total = n_elements / stride * dims;

	float prediction = (float)predictions[i];

	float r = (float)predictions[i - intra_elem_idx + 0];
	float g = (float)predictions[i - intra_elem_idx + 1];
	float b = (float)predictions[i - intra_elem_idx + 2];


	const float Fac = 1.0;
#define SQRT 0

	if(LOD == 0){
		r = clamp(r, 0.0f, 10000.0f);
		g = clamp(g, 0.0f, 10000.0f);
		b = clamp(b, 0.0f, 10000.0f);
	}

	float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

#if !SQRT
	luminance = luminance;
#endif
	//const float prediction_sq_plus_epsilon =sqrt(sqrt(sqrt(sqrt(luminance+0.000001f)+0.0000001f)+ 0.0000001f ) + 0.0000001f )+ 0.001f;

	
	if (LOD == 0) {
		data_pdf[target_idx] = luminance;
		
	}
	else {
		luminance += data_pdf[target_idx];
		luminance = clamp(luminance, 0.0f, 1000000.0f);
	}
	const float prediction_sq_plus_epsilon =luminance + 0.01f;
	//const float prediction_sq_plus_epsilon = 1.0f;

	const float pdf =  1;

	const float target_value = targets[target_idx];

#define EMA 1

#if EMA
	prediction = prediction*ema+(1.0-ema)*target_value;
#endif

#if SQRT
	const float difference = sqrt(clamp(prediction*Fac, 0.0f, 10000.0f) + 0.000001) - sqrt(target_value*Fac + 0.000001) / pdf;
#else
	const float difference = prediction - target_value / pdf;
#endif

	values[i] = difference * difference / prediction_sq_plus_epsilon / n_total;

	float gradient = loss_scale * (difference / prediction_sq_plus_epsilon)/n_total;
	/*if (isnan(gradient) || abs(gradient) > 1000.0f) {
		gradient = fminf(fmaxf(gradient, -1000.0f), 1000.0f);
	}*/


	gradients[i] = (T)(gradient);
	if(LOD == 0)
		targets[target_idx] = -difference;
	
}


template <typename T>
class RelativeL2LuminanceLoss : public Loss<T> {
public:
	RelativeL2LuminanceLoss(const json& lossData){
		m_lod = 0;
		m_ema = 1.0f;
		update_hyperparams(lossData);
	}

	RelativeL2LuminanceLoss(){
		m_lod = 0;
		m_ema = 1.0f;
	}

	void evaluate(
		cudaStream_t stream,
		const uint32_t stride,
		const uint32_t dims,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr) const override {
		if (prediction.n() != target.n()) {
			throw std::runtime_error(std::string("Prediction and target don't have matching batch size ") + std::to_string(prediction.n()) + "!=" + std::to_string(target.n()));
		}

		if (prediction.m() != stride) {
			throw std::runtime_error(std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m()) + "!=" + std::to_string(stride));
		}

		if (target.m() != dims) {
			throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims));
		}

		linear_kernel(relative_l2_luminance_loss<T>, 0, stream,
			prediction.n_elements(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? data_pdf->data() : nullptr
		);
	}
	
	void evaluate(
		cudaStream_t stream,
		const uint32_t stride,
		const uint32_t dims,
		uint32_t batch_size,
		int* mapping,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr) const override {
		/*if (prediction.n() != batch_size) {
			throw std::runtime_error(std::string("Prediction and mapping don't have matching batch size ") + std::to_string(prediction.n()) + "!=" + std::to_string(target.n()));
		}*/

		if (prediction.m() != stride) {
			throw std::runtime_error(std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m()) + "!=" + std::to_string(stride));
		}

		if (target.m() != dims) {
			throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims));
		}

		linear_kernel(relative_l2_luminance_loss_lod<T>, 0, stream,
			prediction.n_elements(),
			batch_size* prediction.m(),
			mapping,
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? reinterpret_cast<float*>(data_pdf->data_mutable()) : nullptr,
			m_lod,
			m_ema
		);
	}

	void update_hyperparams(const json& params) override { 
		if (params.contains("lod")) {
			m_lod = params["lod"];
		}

		if (params.contains("ema")) {
			m_ema = params["ema"];
		}

	}
protected:
	uint32_t m_lod;
	float m_ema;
};

TCNN_NAMESPACE_END
