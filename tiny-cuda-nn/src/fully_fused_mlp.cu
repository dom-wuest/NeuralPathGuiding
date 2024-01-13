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

/** @file   fully_fused_mlp.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>
#include <tiny-cuda-nn/common_device.h>

#include <mma.h>


TCNN_NAMESPACE_BEGIN

void check_shmem_error(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, weights_layout_t> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__syncthreads();

	// Load N_BLOCKS chunks of weights from global memory into registers.
	#pragma unroll
	for (uint32_t i = 0; i < N_BLOCKS; ++i) {
		if (BACKWARD) {
			// If we're performing the backward pass, additional index swizzling is needed to
			// load the weights in transposed form.
			wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
		} else {
			wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
		}
	}

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::fill_fragment(result_frag[l], 0.0f);

		#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
		}

		// Activation
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			wmma::load_matrix_sync(act_frag, activation_aux + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * WIDTH, WIDTH);
			warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
		} else {
			warp_activation<__half>(activation, result_frag[l], result_frag[l]);
		}
	}

	__syncthreads();

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	#pragma unroll
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH];
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
__global__ void kernel_mlp_fused_backward(const __half* __restrict__ dL_doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, __half* __restrict__ dL_dinput, const __half* __restrict__ weights_first_layer, const uint32_t batch_size, const uint32_t out_width, const uint32_t n_hidden_matmuls) {
	// `dL_doutput` points to the input matrix of the backward pass, i.e. the loss gradients. Assumed to be 16 neurons wide.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where backpropagated activation gradients should be written.
	// `forward` points to the memory where the intermediate activations of the forward pass are located. (needed for activation backprop)

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")
	const uint32_t bi = blockIdx.x;	 // block index

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// A skew is applied to the matrix storage to avoid bank conflicts.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	// Multipying one 16-row chunk of intermediate activations with the weight matrix requires all warps of the block.
	// Thus, each block computes exactly one 16-row chunk of the next layer's intermediate activations.
	const uint32_t elem_idx_base = 16 * bi * N_ITERS * BLOCK_DIM_Z;
	const uint32_t elem_idx = elem_idx_base + 16 * threadIdx.z;

	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;

	// Backprop through last layer
	if (out_width <= 16) {
		using namespace nvcuda;

		// Fragments in registers
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> weights_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

		// Load the relevant chunk of the last layer's weight matrix from global memory into registers
		const uint32_t weights_col = 16 * wi;

		wmma::load_matrix_sync(weights_frag, weights + layer_stride * n_hidden_matmuls + weights_col, WIDTH);

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::fill_fragment(result_frag[l], 0.0f);

			// Load a chunk of output gradients from shared memory and multiply with previously loaded weights
			if (std::is_same<OUTPUT_LAYOUT, wmma::row_major>::value) {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * 16, 16);
			} else {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)), batch_size);
			}

			// NOTE: activation transfer of the _output_ activation is expected to be done _prior_ to calling this kernel
			//       in a separate pass, because the tranfered activation gradient is also needed to compute the weight
			//       gradient of the last weight matrix (see backward()).
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

			// Load the temporary forward matrix for the relu transfer
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> forward_frag;
			wmma::load_matrix_sync(forward_frag, forward + output_stride * n_hidden_matmuls + weights_col + (elem_idx + l * BLOCK_DIM_Z * 16) * WIDTH, WIDTH);

			warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
		}

		__syncthreads();

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
		}

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate[lane_offset + (row + elem_idx + i * BLOCK_DIM_Z * 16) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	} else {
		// If the output width is larger than 16, we will have used CUTLASS for backpropping through the last layer.
		// Load the resulting gradients.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
	}

	// Backprop through hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + layer_stride * (n_hidden_matmuls - k - 1), out_intermediate + output_stride * (k + 1) + elem_idx_base * WIDTH, forward + output_stride * (n_hidden_matmuls - k - 1) + elem_idx_base * WIDTH);
	}

	// Compute loss gradients w.r.t. input if desired.
	// THIS CODE ASSUMES THAT THE INPUT WIDTH IS THE SAME AS THE NETWORK WIDTH.
	// DON'T PASS A NON-NULL dL_dinput IF THIS REQUIREMENT IS NOT MET.
	if (dL_dinput != nullptr) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
	}
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_backward(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrix<T>* dL_dinput,
	const uint32_t n_hidden_matmuls
) {
	throw std::runtime_error{"The fully fused backward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_backward(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrix<T>* dL_dinput,
	const uint32_t n_hidden_matmuls
) {
	const uint32_t batch_size = dL_doutput.cols();
	const uint32_t out_width = dL_doutput.rows();
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	if (forward.cols() != batch_size) {
		throw std::runtime_error{"Batch size of matrices dL_doutput and temporaries doesn't match."};
	}

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = 1;

	if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) {
		throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};
	}

	const dim3 threads = { 32u, N_BLOCKS, BLOCK_DIM_Z }; // 32 threads = 1 warp, 8 warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	int shmem_size = sizeof(__half) * ((16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights
	const dim3 blocks = { n_blocks, 1u, 1u };

	// The kernels operate with transposed layouts compared with the MLP code
	if (dL_doutput.layout() == RM) {
		check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), batch_size, out_width, n_hidden_matmuls);
	} else {
		check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), batch_size, out_width, n_hidden_matmuls);
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// input_threadblock points to the thread block's chunk of the input batch in global memory
	// weights_this_layer points to the weight matrix of the current layer
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// in_width is the dynamic width of the input layer

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + BLOCK_DIM_Z * 16 * (in_width + INPUT_SKEW);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS*BLOCK_DIM_Z warps
	const uint32_t n_elems_per_load = N_BLOCKS * 32 * BLOCK_DIM_Z * 8;
	const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8;

	const uint32_t n_elems_b = WIDTH * in_width;

	#pragma unroll
	for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
		const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
	}

	const uint32_t n_tensor_ops = in_width / 16;

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		// Load chunk of inputs into shmem.
		// This is faster than loading it from gmem directly, even though it is only used once.
		// (Possibly due to latency hiding through staging.)
		const uint32_t n_elems_a = BLOCK_DIM_Z * 16 * in_width;

		#pragma unroll
		for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
			const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
			*(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
		}

		__syncthreads();

		wmma::fill_fragment(result_frag[l], 0.0f);
		#pragma unroll
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// Load chunk of inputs and weights from shared memory and multiply them
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * threadIdx.z) * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
		}

		__syncthreads();

		warp_activation<__half>(activation, result_frag[l], result_frag[l]);
	}

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// weights_this_layer points to the weight matrix of the current layer
	// out points to the location where the result produced by the thread block should be written to.
	//   Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * BLOCK_DIM_Z * 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

	// Load weight matrix into shared memory for the last multiplication.
	// Loading into shared memory as opposed to directly into registers is faster
	// because unlike in the previous layers, each warp uses the same entries of the weight matrix.
	if (threadIdx.z == 0) {
		*(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
	}

	__syncthreads();

	#pragma unroll
	for (uint32_t i = 0; i < N_BLOCKS; ++i)
		wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

	const bool isUniformActivation = activation != Activation::VMF;

	// Perform last layer by parallelizing over iters
	for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
		wmma::fill_fragment(result_frag, 0.0f);
		#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + idx * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
		}

		//Doesnt neeed for nonuniform as we have a separate one
		if (isUniformActivation) {
			warp_activation<__half>(activation, result_frag, result_frag);
		}

		if (output_layout == wmma::mem_row_major) {
			wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
		} else {
			wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	__syncthreads();

	#pragma unroll
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const nvcuda::wmma::layout_t output_layout = nvcuda::wmma::mem_row_major) {
	// `input` points to the input matrix. Can be any width.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
	// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

	// Commented out due to isolated strange side-effects on Windows
	// if (INFERENCE) {
	// 	assert(out_intermediate == nullptr);
	// } else {
	// 	assert(out_intermediate);
	// }

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// In some cases, it also contains the weight matrix for the first and last layer.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z;

	// First layer
	if (in_width == WIDTH) {
		// If the input has the same width as the network, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	} else {
		threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width);
	}

	const uint32_t first_layer_size = WIDTH * in_width;
	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;

	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, !INFERENCE ? (out_intermediate + output_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
	}

	if (out_width > 16) {
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		// Last layer
		if (output_layout == nvcuda::wmma::mem_row_major) {
			threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx * 16, 16, output_layout);
		} else {
			threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx, batch_size, output_layout);
		}
	}
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	throw std::runtime_error{"The fully fused forward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	const uint32_t batch_size = input.cols();
	const uint32_t in_width = input.rows();

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
	if (in_width % 16 != 0) {
		throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
	}

	if (weights.rows() != WIDTH) {
		throw std::runtime_error{"The fully fused forward pass only works with WIDTH-sized matrices."};
	}

	if (weights.cols() % 16 != 0) {
		throw std::runtime_error{std::string("weights must have a multiple-of-16 number of columns. ") + std::to_string(weights.cols())};
	}

	if (output_intermediate.cols() != batch_size) {
		throw std::runtime_error{"Batch size of inputs and output_intermediate doesn't match."};
	}

	if (output && output->cols() != batch_size) {
		throw std::runtime_error{"Batch size of inputs and outputs doesn't match."};
	}

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = (INFERENCE && WIDTH == 128) ? 2 : 1;

	if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) {
		throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};
	}

	const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
	if (in_width != WIDTH) {
		// If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
		shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (in_width + INPUT_SKEW));
	}

	const dim3 blocks = { n_blocks, 1u, 1u };

	check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
	kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
		output_activation,
		input.data(),
		weights.data(),
		output_intermediate.data(),
		output ? output->data() : nullptr,
		batch_size,
		in_width,
		output ? output->rows() : 0,
		n_hidden_layers,
		output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major // The kernels operate with transposed layouts compared with the MLP code
	);
}

template <typename T, int WIDTH>
FullyFusedMLP<T, WIDTH>::FullyFusedMLP(
	uint32_t input_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	bool use_feedback_alignment,
	Activation activation,
	Activation output_activation
) :
m_input_width{input_width},
m_network_width{WIDTH},
m_output_width{output_width},
m_n_hidden_layers{n_hidden_layers},
m_use_feedback_alignment{use_feedback_alignment},
m_activation{activation},
m_output_activation{output_activation}
{
	if (m_n_hidden_layers <= 0) {
		throw std::runtime_error("FullyFusedMLP requires at least 1 hidden layer (3 layers in total).");
	}

	m_n_hidden_matmuls = n_hidden_layers-1;

	m_padded_output_width = next_multiple(m_output_width, tensorcore_width);

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_backward.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_input_width);
	m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_backward.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_backward.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}

	// Buffers to keep data from the forward and backward pass
	m_forward_tmp.resize(m_n_hidden_layers);
	m_backward_tmp.resize(m_n_hidden_layers);

	// 1 stream per matmul
	m_training_splitk_streams.resize(m_n_hidden_layers + 1);
	m_training_splitk_events.resize(m_n_hidden_layers + 1);

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T, int WIDTH>
FullyFusedMLP<T, WIDTH>::~FullyFusedMLP() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		cutlass_free_workspace(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrix<float>& output) {
	inference_mixed_precision(stream, input, m_inference_output_tmp);

	//const uint32_t n_elements = (uint32_t)output.n_elements();
	const uint32_t n_elements = (uint32_t)output.n_elements();
	if( m_output_activation == Activation::VMF)
		apply_vmf<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data());
	trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrix<float>& output, 
					uint32_t size, int* mapping, float* bbStart, float* bbEnd) {
	inference_mixed_precision(stream, input, m_inference_output_tmp);

	//const uint32_t n_elements = (uint32_t)output.n_elements();


	if(false){
		const uint32_t n_elements = (uint32_t)output.n_elements();
		if( m_output_activation == Activation::VMF)
			apply_vmf<T> << <n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data());
		trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
	}
	else{
		trim_and_cast<T><<<n_blocks_linear(size*m_output_width), n_threads_linear, 0, stream>>>(size* m_output_width, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data(), mapping, bbStart, bbEnd);
	}


	
}

template <typename CutlassLayer, MatrixLayout input_layout, typename T>
void compute_inference_layer(
	cudaStream_t stream,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, input_layout>& input,
	GPUMatrixDynamic<T>& output
) {
	fc_multiply<CutlassLayer>(stream, weights, input, output, activation);
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference_mixed_precision(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (&output != &m_inference_output_tmp && output.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(m_output_width));
	}

	if (&output != &m_inference_output_tmp && input.n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if(m_inference_tmp.n() != batch_size)
	allocate_inference_buffers(batch_size, true);


	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Forward;

	// ASSUMPTION: weight matrices are contiguous in memory
	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, true>(       stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(    stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, true>(       stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>( stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(   stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (m_output_width > 16) {
		compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(weight_usage), m_inference_tmp, output);
	}
}
template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, const GPUMatrix<T>& input, uint32_t size, int* mapping, float* bbStart, float* bbEnd, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) {
	forward(stream, input, output, use_inference_matrices, prepare_input_gradients);
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_matrices, bool prepare_input_gradients) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (output && output->m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width (must be padded): ") + std::to_string(output->m()) + "!=" + std::to_string(m_padded_output_width));
	}

	if (output && input.n() != output->n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output->n()));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if(m_forward_tmp.front().n() != batch_size)
		allocate_forward_buffers(batch_size, true);

	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Forward;

	// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, false>(       stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, false>(    stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, false>(       stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, false>( stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, false>(   stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		case Activation::VMF:         mlp_fused_forward<WIDTH, T, Activation::VMF, false>(        stream, m_output_activation, input_weight_matrix(weight_usage), input, m_forward_tmp.at(0), output, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (output && m_output_width > 16) {
		compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(weight_usage), m_forward_tmp.back(), *output);
	}

	const uint32_t n_elements = (uint32_t)output->n_elements();

	const bool isUniformActivation = m_output_activation != Activation::VMF;
	if (!isUniformActivation) {
		apply_vmf<T> << <n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, output->data());
	}
}


template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::backward(
	cudaStream_t stream,
	const GPUMatrix<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>* dL_dinput,
	bool use_inference_matrices,
	bool compute_param_gradients
) {
	if (dL_doutput.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output gradients have incorrect width (must be padded): ") + std::to_string(dL_doutput.m()) + "!=" + std::to_string(m_padded_output_width));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();
	if (m_backward_tmp.front().n() != batch_size) {
		allocate_backward_buffers(batch_size);
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	if (m_output_activation != Activation::None) {
		if(m_output_activation == Activation::VMF){
			const uint32_t n_elements = (uint32_t)output.n_elements();
			vmf_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data(), m_padded_output_width, VMF_COUNT*4);
		}
		else
			activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), m_backward_output_tmp.data());
	}
	// Backprop
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Backward;

	//uint32_t old_forward_size = m_forward_tmp[0].cols();
	//// CRUNCH
	//for (uint32_t i = 0; i < m_forward_tmp.size(); i++)
	//	m_forward_tmp[i].set_size_const(m_forward_tmp[i].rows(), batch_size);


	{
		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		m_backward_output_tmp.set_layout(dL_doutput.layout());
		const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : m_backward_output_tmp;

		uint32_t tmp_idx = m_n_hidden_matmuls;
		uint32_t backward_tmp_idx = 0;

		if (compute_param_gradients) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);

			// Compute weight gradients
			fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), tmp_dL_doutput, m_forward_tmp.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor);

			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If the output width is larger than 16 dims, we use cutlass to backpropagate through the last layer
		// rather than fusing it with our kernel.
		if (m_output_width > 16) {
			fc_multiply<FullLayer>(stream, output_weight_matrix(weight_usage).transposed(), tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx), m_activation, true);
		}

		// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
		auto dL_dinput_fused = input.m() == m_forward_tmp.at(0).m() ? dL_dinput : nullptr; // Only let the fully fused kernel compute gradients w.r.t. the input, if the input layer has the same size as the other layers

		switch (m_activation) {
			case Activation::None:        mlp_fused_backward<WIDTH, T, Activation::None>(       stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			case Activation::Exponential: mlp_fused_backward<WIDTH, T, Activation::Exponential>(stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			case Activation::Sigmoid:     mlp_fused_backward<WIDTH, T, Activation::Sigmoid>(    stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			case Activation::ReLU:        mlp_fused_backward<WIDTH, T, Activation::ReLU>(       stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			case Activation::Squareplus:  mlp_fused_backward<WIDTH, T, Activation::Squareplus>( stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			case Activation::Softplus:    mlp_fused_backward<WIDTH, T, Activation::Softplus>(   stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
			default: throw std::runtime_error{"Unsupported activation."};
		}

		tmp_idx -= 1;
		++backward_tmp_idx;

		// layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

			if (compute_param_gradients) {
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), m_forward_tmp.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor);
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
			}

			tmp_idx -= 1;
			++backward_tmp_idx;
		}

		if (compute_param_gradients) {
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
			fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor);
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If requested and if the fully fused kernel didn't already take care of it, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput && input.m() != m_forward_tmp.at(0).m()) {
			// TODO: optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<FullLayer>(stream, input_weight_matrix(weight_usage).transposed(), m_backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
		}
	}

	//for (uint32_t i = 0; i < m_forward_tmp.size(); i++)
	//	m_forward_tmp[i].set_size_const(m_forward_tmp[i].rows(), old_forward_size);

	if (compute_param_gradients) {
		// All the per-layer split-k matrix multiplications summing over
		// the batch are computed in parallel streams to the actual
		// backpropagation. Here, we need to wait for all of these to complete.
		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_inference_buffers(uint32_t batch_size, bool bAllocate=true) {
	m_inference_tmp.set_size(m_network_width, batch_size);
	m_inference_output_tmp.set_size(m_padded_output_width, batch_size);

	if(bAllocate){
		GPUMatrixBase::allocate_shared_memory(
			m_inference_buffer,
			{
				&m_inference_tmp,
				&m_inference_output_tmp,
			}
		);
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_forward_buffers(uint32_t batch_size, bool bAllocate=true) {
	for (size_t i = 0; i < m_forward_tmp.size(); ++i) {
		m_forward_tmp[i].set_size(m_network_width, batch_size);
	}

	if(bAllocate){
		GPUMatrixBase::allocate_shared_memory(m_forward_buffer, m_forward_tmp);
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_backward_buffers(uint32_t batch_size) {
	std::vector<GPUMatrixBase*> matrix_pointers = {&m_backward_output_tmp};

	m_backward_output_tmp.set_size(m_padded_output_width, batch_size);
	for (uint32_t i = 0; i < (uint32_t)m_backward_tmp.size(); ++i) {
		m_backward_tmp[i].set_size(m_network_width, batch_size);
		matrix_pointers.emplace_back(&m_backward_tmp[i]);
	}

	GPUMatrixBase::allocate_shared_memory(m_backward_buffer, matrix_pointers);
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data(params + current_pos);
		m_weight_matrices_inference[i].set_data(inference_params + current_pos);
		m_weight_matrices_backward[i].set_data((m_use_feedback_alignment ? backward_params : params) + current_pos);
		m_weight_matrices_full_precision[i].set_data(params_full_precision + current_pos);
		m_gradient_matrices[i].set_data(gradients + current_pos);

		current_pos += m_weight_matrices[i].n_elements();
	}

	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				m_weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else if (m_use_feedback_alignment) {
			m_weight_matrices_full_precision[i].initialize_fa_uniform_forward(rnd, scale);
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}

	// Initialize backward params for feedback alignment
	if (m_use_feedback_alignment) {
		for (size_t i = 0; i < m_weight_matrices_backward.size(); ++i) {
			m_weight_matrices_backward[i].initialize_fa_uniform_backward(rnd, scale);
		}
	}
}

template class FullyFusedMLP<network_precision_t, 256>;
template class FullyFusedMLP<network_precision_t, 128>;
template class FullyFusedMLP<network_precision_t, 64>;
template class FullyFusedMLP<network_precision_t, 32>;
template class FullyFusedMLP<network_precision_t, 16>;

TCNN_NAMESPACE_END
