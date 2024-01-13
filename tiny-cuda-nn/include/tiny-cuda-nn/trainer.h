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

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/gpu_memory_json.h>

#include <iostream>
#include <random>


TCNN_NAMESPACE_BEGIN

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Trainer : public ObjectWithMutableHyperparams {
public:
	Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
	: m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
		std::seed_seq seq{seed};
		std::vector<uint32_t> seeds(2);
		seq.generate(std::begin(seeds), std::end(seeds));
		m_rng = pcg32{seeds.front()};
		initialize_params();
	}

	virtual ~Trainer() {}

	void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
		if (!loss) {
			throw std::runtime_error{"Trainer: may not set loss to nullptr"};
		}
		m_loss = loss;
	}

	void initialize_params() {
		size_t n_params = m_model->n_params();
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "Trainer: Initializing " << n_params << " params and resetting training." << std::endl;
#endif

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 3 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		m_params_full_precision = (float*)(m_params_buffer.data());
		m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
		m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
		m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params * 2);

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		// Use the optimizer's custom params for inference, if they exist.
		m_params_inference = m_optimizer->custom_weights();
		if (m_params_inference == nullptr) {
			m_params_inference = m_params;
		}

		m_model->initialize_params(
			m_rng,
			m_params_full_precision,
			m_params,
			m_params_inference,
			m_params_backward,
			m_param_gradients
		);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params=m_params] __device__ (size_t i) {
			params[i] = (PARAMS_T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void allocate_training_buffers(uint32_t padded_output_width, uint32_t batch_size, bool bAllocate=true) {
		m_perturbation.set_size(padded_output_width, batch_size);
		m_perturbed_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_gradient_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_tmp.set_size(padded_output_width, batch_size);

		if(bAllocate)
		GPUMatrixBase::allocate_shared_memory(
			m_training_buffer,
			{
				&m_perturbation,
				&m_perturbed_training_prediction_tmp,
				&m_training_prediction_tmp,
				&m_training_loss_gradient_tmp,
				&m_training_loss_tmp,
			}
		);
	}

	void set_training_buffers(uint32_t padded_output_width, uint32_t batch_size) {
		m_perturbation.set_size(padded_output_width, batch_size);
		m_perturbed_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_gradient_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_tmp.set_size(padded_output_width, batch_size);
	}

	const GPUMatrix<COMPUTE_T>& forward(cudaStream_t stream, const GPUMatrix<T>& input) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_training_prediction_tmp.n() != batch_size) {
			allocate_training_buffers(m_model->padded_output_width(), batch_size);
		}

		m_model->forward(stream, input, &m_training_prediction_tmp);
		return m_training_prediction_tmp;
	}
	
	const GPUMatrix<COMPUTE_T>& forward(cudaStream_t stream, const GPUMatrix<T>& input,
										uint32_t padded_size, int* mapping, float* bbStart, float* bbEnd) 
	{
		// Make sure our teporary buffers have the correct size for the given batch size
		if(m_training_prediction_tmp.n() != (padded_size+127)/128*128)
		allocate_training_buffers(m_model->padded_output_width(), (padded_size + 127) / 128 * 128, true);

		m_model->forward(stream, input, padded_size, mapping, bbStart, bbEnd, &m_training_prediction_tmp);
		return m_training_prediction_tmp;
	}

	const GPUMatrix<COMPUTE_T>& forward(const GPUMatrix<T>& input) {
		return forward(nullptr, input);
	}

	const GPUMatrix<float>& evaluate_loss(cudaStream_t stream, const float loss_scale, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, float* loss_value = nullptr) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = target.n();
		if (m_training_prediction_tmp.n() != batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` before calling `evaluate_loss`"};
		}

		if (m_perturbation_sigma > 0) {
			const uint32_t n_elements = m_perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, m_perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_training_prediction_tmp.data(), m_perturbation.data(), m_perturbed_training_prediction_tmp.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? m_perturbed_training_prediction_tmp : m_training_prediction_tmp;

		m_loss->evaluate(
			stream,
			m_model->padded_output_width(),
			m_model->output_width(),
			loss_scale,
			loss_input,
			target,
			m_training_loss_tmp,
			m_training_loss_gradient_tmp,
			data_pdf

		);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}

		return m_training_loss_tmp;
	}
	
	const GPUMatrix<float>& evaluate_loss(
		cudaStream_t stream, 
		const float loss_scale, 
		GPUMatrix<float>& target, 
		uint32_t batch_size, int* mapping,
		const GPUMatrix<float>* data_pdf = nullptr, 
		float* loss_value = nullptr) 
	{
		// Make sure our teporary buffers have the correct size for the given batch size
		/*if (m_training_prediction_tmp.n() <= batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` before calling `evaluate_loss`"};
		}*/

		if (m_perturbation_sigma > 0) {
			const uint32_t n_elements = m_perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, m_perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_training_prediction_tmp.data(), m_perturbation.data(), m_perturbed_training_prediction_tmp.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? m_perturbed_training_prediction_tmp : m_training_prediction_tmp;

		m_loss->evaluate(
			stream,
			m_model->padded_output_width(),
			m_model->output_width(),
			batch_size, mapping,
			loss_scale,
			loss_input,
			target,
			m_training_loss_tmp,
			m_training_loss_gradient_tmp,
			data_pdf
		);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}

		return m_training_loss_tmp;
	}

	void evaluate_loss(const float loss_scale, GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, float* loss_value = nullptr) {
		evaluate_loss(nullptr, loss_scale, target, data_pdf, loss_value);
	}

	void backward(cudaStream_t stream, const GPUMatrix<T>& input) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
	/*	if (m_training_prediction_tmp.n() != batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` and `evaluate_loss` before calling `backward`"};
		}*/

		m_model->backward(stream, input, m_training_prediction_tmp, m_training_loss_gradient_tmp);
	}

	void backward(const GPUMatrix<T>& input) {
		backward(nullptr, input);
	}

	void optimizer_step(cudaStream_t stream, float loss_scale) {
		m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
	}

	void optimizer_step(float loss_scale) {
		optimizer_step(nullptr, loss_scale);
	}

	void training_step(
		cudaStream_t stream,
		 GPUMatrix<T>& input,
		GPUMatrix<float>& target,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr
	) {
		if (input.n() != target.n()) {
			throw std::runtime_error(std::string("Input and target don't have matching batch size ") + std::to_string(input.n()) + "!=" + std::to_string(target.n()));
		}
		
		// Because of VMF with different size of training data and model output we need to comment it
		/*if (target.m() != m_model->output_width()) {
			throw std::runtime_error(std::string("Target does not have the correct number of dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(m_model->output_width()));
		}*/

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		bool did_allocate = false;
		if (m_training_prediction_tmp.n() != batch_size) {
			allocate_training_buffers(m_model->padded_output_width(), batch_size);
			did_allocate = true;
		}


		static const float loss_scale = 128;

		m_graph.capture_and_execute(stream, did_allocate, [&]() {
			forward(stream, input);
			evaluate_loss(stream, loss_scale, target, data_pdf);
			backward(stream, input);
		});

		optimizer_step(stream, loss_scale);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}
	}
	
	void training_step(
		cudaStream_t stream,
		 GPUMatrix<T>& input,
		GPUMatrix<float>& target,
		uint32_t batch_size, int* mapping, float* bbStart, float* bbEnd,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool bTrain = true
	) {
		if (input.n() != target.n()) {
			throw std::runtime_error(std::string("Input and target don't have matching size") + std::to_string(input.n()) + "!=" + std::to_string(target.n()));
		}

		const uint32_t paddedSize = (batch_size+127) / 128 * 128;
		const uint32_t oldInputSize = input.n();
		input.set_size(input.m(), paddedSize);
		target.set_size(target.m(), paddedSize);
			// Because of VMF with different size of training data and model output we need to comment it
		if (target.m() != m_model->output_width()) {
			throw std::runtime_error(std::string("Target does not have the correct number of dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(m_model->output_width()));
		}

		// Make sure our teporary buffers have the correct size for the given batch size
		bool did_allocate = m_training_prediction_tmp.n() != paddedSize;
		if(did_allocate)
		allocate_training_buffers(m_model->padded_output_width(), paddedSize, did_allocate);

		static const float loss_scale = 128;

		m_graph.capture_and_execute(stream, did_allocate, [&]() {
			forward(stream, input, batch_size, mapping, bbStart, bbEnd);
			evaluate_loss(stream, loss_scale, target, batch_size, mapping, data_pdf);
			backward(stream, input);
		});

		if(bTrain)
			optimizer_step(stream, loss_scale);
		input.set_size(input.m(), oldInputSize);
		target.set_size(target.m(), oldInputSize);
		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}

		/*if (isnan(*loss_value)) {
			std::cout << "NNNAANAN";
		}*/

	/*	input.set_size(input.m(), batch_size);
		target.set_size(target.m(), batch_size);*/
	}

	void training_step(
		 GPUMatrix<T>& input,
		const GPUMatrix<float>& target,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr
	) {
		training_step(nullptr, input, target, loss_value, data_pdf);
	}

	void update_hyperparams(const json& params) override {
		m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
		m_loss->update_hyperparams(params.value("loss", json::object()));
	}

	float* params() {
		return m_params_full_precision;
	}

	void set_params_full_precision(const float* params_cpu, size_t n_params) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because CPU buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_full_precision, params_cpu, sizeof(float)*n_params, cudaMemcpyHostToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_inference[i] = (PARAMS_T)params_fp[i];
		});

		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_params(const PARAMS_T* params_cpu, size_t n_params) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because CPU buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_inference, params_cpu, sizeof(PARAMS_T)*n_params, cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_fp[i] = (float)params_inference[i];
		});

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
		return m_model;
	}

	json serialize(bool serialize_optimizer = false) {
		size_t n_params = m_model->n_params();

		json data;
		data["n_params"] = n_params;
		data["params_binary"] = gpu_memory_to_json_binary(m_params_inference, sizeof(PARAMS_T)*n_params);

		if (serialize_optimizer) {
			data["optimizer"] = m_optimizer->serialize();
		}

		return data;
	}

	void deserialize(const json& data) {
		json::binary_t params_binary = data["params_binary"];
		set_params((PARAMS_T*)params_binary.data(), params_binary.size()/sizeof(PARAMS_T));

		if (data.contains("optimizer")) {
			m_optimizer->deserialize(data["optimizer"]);
		}

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
	std::shared_ptr<Loss<COMPUTE_T>> m_loss;

	CudaGraph m_graph;

	GPUMemory<char> m_params_buffer;

	float* m_params_full_precision = nullptr;
	PARAMS_T* m_params_inference = nullptr;
	PARAMS_T* m_params = nullptr;
	PARAMS_T* m_params_backward = nullptr; // Used for wonky things like feedback alignment
	PARAMS_T* m_param_gradients = nullptr;

	float m_perturbation_sigma;

	GPUMemory<char> m_training_buffer;

	GPUMatrix<float> m_perturbation;
	GPUMatrix<COMPUTE_T> m_perturbed_training_prediction_tmp;
	GPUMatrix<COMPUTE_T> m_training_prediction_tmp;
	GPUMatrix<COMPUTE_T> m_training_loss_gradient_tmp;
	GPUMatrix<float> m_training_loss_tmp;

	pcg32 m_rng;
};

TCNN_NAMESPACE_END
