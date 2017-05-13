#include <random>
#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>


#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>


#define recur_batch_size 2;



#define TRAINING (false)

#ifndef PERFOPTS
#define PERFOPTS (31)
#endif



int option = 0;

#define GROUP_GEMM ((PERFOPTS & 1))
#define USE_STREAMS true
#define FUSE_PW ((PERFOPTS & 4))
#define PRE_TRANSPOSE ((PERFOPTS & 8))
#define RECUR_BATCH_SIZE (((PERFOPTS & 16) ? 2 : 1))

// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
	return 1.f / (1.f + expf(-in));
}


__global__ void elementWise_fp(int hiddenSize, int miniBatch,
							   float *tmp_h,
							   float *tmp_i,
							   float *bias,
							   float *linearGates,
							   float *h_out,
							   float *i_out,
							   float *c_in,
							   float *c_out,
							   bool training) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int numElements = miniBatch * hiddenSize;

	if (index >= numElements) return;

	int batch = index / hiddenSize;
	int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

	float g[4];

	for (int i = 0; i < 4; i++) {
		g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
		g[i] += bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];

		if (training) linearGates[gateIndex + i * hiddenSize] = g[i];
	}


	float in_gate     = sigmoidf(g[0]);
	float forget_gate = sigmoidf(g[1]);
	float in_gate2    = tanhf(g[2]);
	float out_gate    = sigmoidf(g[3]);

	float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);

	c_out[index] = val;

	val = out_gate * tanhf(val);

	h_out[index] = val;
	i_out[index] = val;
}


// define the error information

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

float one = 1.f;
float zero = 0.f;


class LSTMNetwork{
public:
	cublasHandle_t handle;


	// arguments about the network
	int num_layers;
	int mem_cell_num;
	int input_dim;
	int mini_batch;
	int seq_length;
	int num_elements;

	float *input_data;
	float *output_data;
	float *cell_state;

	// workspace for the result of R * h and W * x
	float *temp_output;
	float *temp_input;

	// used for training
	float *activations;


	// stream and event
	cudaStream_t *stream_i;
	cudaStream_t *stream_h;

	cudaEvent_t **events_i;
	cudaEvent_t **events_h;

	bool training;

	// W and R
	// W is for the input
	// R is for the h_prev
	// TODO: is it necessary to have another work space for weight 
	float *weight_in;
	float *weight_out;

	float *bias;


	LSTMNetwork(int num_layers, int mem_cell_num, int input_dim, int mini_batch, int seq_length, float *input);
    ~LSTMNetwork();

	float feedforward(bool training);
	void backprop();
	void transpose_weight();

};


LSTMNetwork::LSTMNetwork(int num_layers, int mem_cell_num, int input_dim, int mini_batch, int seq_length, float *input) {
	/*this->mini_batch = 64;
	bool checkF = true;
	this->num_layers = 4;
	this->seq_length = 100;
	this->mem_cell_num = 512;
	this->input_dim = 512;*/

	this->num_layers = num_layers;
	this->mem_cell_num = mem_cell_num;
	this->input_dim = input_dim;
	this->mini_batch = mini_batch;
	this->seq_length = seq_length;

	// initialize the handle 
	cublasErrCheck(cublasCreate(&this->handle));

	this->num_elements = input_dim * mini_batch;

	// initalize stream and event;
	stream_i = (cudaStream_t*)malloc(num_layers * sizeof(cudaStream_t));
	stream_h = (cudaStream_t*)malloc(num_layers * sizeof(cudaStream_t));

	events_i = (cudaEvent_t**)malloc(num_layers * sizeof(cudaEvent_t*));
	events_h = (cudaEvent_t**)malloc(num_layers * sizeof(cudaEvent_t*));
	for (int i = 0; i < num_layers; i++) {
		events_i[i] = (cudaEvent_t*)malloc(seq_length * sizeof(cudaEvent_t));
		events_h[i] = (cudaEvent_t*)malloc(seq_length * sizeof(cudaEvent_t));
	}

	// initialize  stream
	for (int i = 0; i < num_layers; i++) {
		if (USE_STREAMS) {
			cudaErrCheck(cudaStreamCreate(&stream_i[i]));
			// Priority is empirical.
			cudaErrCheck(cudaStreamCreateWithPriority(&stream_h[i], 0, -1));
		} else {
			stream_i[i] = NULL;
			stream_h[i] = NULL;
		}
	}

   
	cudaErrCheck(cudaMalloc((void**)&output_data, (seq_length + 1) * (num_layers) * num_elements * sizeof(float)));

	cudaErrCheck(cudaMalloc((void**)&input_data, (seq_length) * (num_layers + 1) * num_elements * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&cell_state, (seq_length + 1) * (num_layers) * num_elements * sizeof(float)));

	cudaErrCheck(cudaMalloc((void**)&weight_in, num_layers * mem_cell_num * input_dim * 8 * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&weight_out, num_layers * mem_cell_num * input_dim * 8 * sizeof(float)));


	cudaErrCheck(cudaMalloc((void**)&temp_output, 4 * num_layers * num_elements * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&temp_input, 4 * seq_length * num_elements * sizeof(float)));
	

	cudaErrCheck(cudaMalloc((void**)&bias, num_layers * mem_cell_num * 8 * sizeof(float)));

	// TODO: copy input into the input_data
	// TOOD: randomlize the first column of each layer for cell_data and output_data




	cudaErrCheck(cudaMalloc((void**)&activations, 4 * seq_length * num_layers * num_elements * sizeof(float)));


	// randomlize the weight 
	curandGenerator_t rng;
	curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
	curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
	curandErrCheck(curandGenerateUniform(rng, this->weight_in, num_layers * mem_cell_num * input_dim * 8));
	curandErrCheck(curandGenerateUniform(rng, this->output_data, (seq_length + 1) * (num_layers) * num_elements));
	curandErrCheck(curandGenerateUniform(rng, this->cell_state, (seq_length + 1) * (num_layers) * num_elements));
	curandErrCheck(curandGenerateUniform(rng, this->input_data, (seq_length) * (num_layers + 1) * num_elements));

	curandErrCheck(curandGenerateUniform(rng, bias, num_layers * mem_cell_num * 8));
	curandErrCheck(curandDestroyGenerator(rng));

	// TOOD: do we need this?
	cudaErrCheck(cudaDeviceSynchronize());

}

LSTMNetwork::~LSTMNetwork() {
    cublasErrCheck(cublasDestroy(this->handle));
	cudaErrCheck(cudaFree(output_data));
	cudaErrCheck(cudaFree(input_data));
	cudaErrCheck(cudaFree(cell_state));

	cudaErrCheck(cudaFree(this->weight_in));
	cudaErrCheck(cudaFree(this->weight_out));
	cudaErrCheck(cudaFree(bias));
	cudaErrCheck(cudaFree(activations));

}


// optimization 4: PRE-TRANSPOSING THE WEIGHT MATRIX
void LSTMNetwork::transpose_weight() {

	for (int i = 0; i < this->num_layers; i++) {
		float *W_in_pointer = this->weight_in + i * this->mem_cell_num * this->input_dim * 8;
		float *W_out_pointer = this->weight_out + i * this->mem_cell_num * this->input_dim * 8;

		float *R_in_pointer = this->weight_in + i * this->mem_cell_num * this->input_dim * 8 + this->mem_cell_num * this->input_dim * 4;
		float *R_out_pointer = this->weight_out + i * this->mem_cell_num * this->input_dim * 8 + this->mem_cell_num * this->input_dim * 4;

		// transpose 4 * W for one layer
		cublasErrCheck(cublasSetStream(handle, stream_i[i]));
		cublasErrCheck(cublasSgeam(this->handle, CUBLAS_OP_T, CUBLAS_OP_N,
			4 * this->mem_cell_num, this->input_dim,
			&one, W_in_pointer, this->mem_cell_num,
			&zero, NULL, 4 * this->mem_cell_num,
			W_out_pointer, 4 * this->mem_cell_num));

		// transpose 4 * R for one layer
		cublasErrCheck(cublasSetStream(handle, stream_h[i]));
		cublasErrCheck(cublasSgeam(this->handle, CUBLAS_OP_T, CUBLAS_OP_N,
			4 * this->mem_cell_num, this->input_dim,
			&one, R_in_pointer, this->mem_cell_num,
			&zero, NULL, 4 * this->mem_cell_num,
			R_out_pointer, 4 * this->mem_cell_num));




		//cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_i_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_i_out, 4 * hiddenSize));


	}

}

// helper function for forward
__global__ void element_wise_operation(int mem_cell_num, int mini_batch,
													float *temp_input, float *temp_output, float *bias,
													float *output_data, float *input_data, float *cell_prev,
													float *cell_curr, bool training, float *activation) {



//	h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
		//	i_data + i * numElements + (layer + 1) * seqLength * numElements,


	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int num_element = mem_cell_num * mini_batch;

	if (index > num_element) {
		return;
	}

	int column_index = index / mem_cell_num;
	int row_index = index % mem_cell_num;

	float result[4];

	// TODO: check the error of this part
	for (int i = 0; i < 4; i++) {
		int real_row_index = row_index + i * mem_cell_num;
		int index_for_temp = column_index * 4 * mem_cell_num + real_row_index;
		result[i] = *(temp_input + index_for_temp) + *(temp_output + index_for_temp)
				+ *(bias + real_row_index) + *(bias + (i + 4) * mem_cell_num + row_index);
	}

	// sequence: i, f, o, ct, other matrix should be stacked in this order
	result[0] = sigmoidf(result[0]);
	result[1] = sigmoidf(result[1]);
	result[2] = sigmoidf(result[2]);
	result[3] = tanhf(result[3]);


	if (training) {
		for (int i = 0; i < 4; i++) {
			activation[column_index * 4 * mem_cell_num + i * mem_cell_num + row_index] = result[i];
		}
	}

	int data_offset = column_index * mem_cell_num + row_index;

	float ct_prev = *(cell_prev + data_offset);
	float ct = result[1] * ct_prev + result[0] * result[3];
	float ht = result[2] * tanhf(ct);

	// store ct and ht
	*(cell_curr + data_offset) = ct;
	*(output_data + data_offset) = ht;
	*(input_data + data_offset) = ht;

}

// perform the feedforward for entire network
float LSTMNetwork::feedforward(bool training) {

	float elapsedTime;
	option = 2;


	if (option == 2) {


		bool checkF = true;

		// Timing starts here

		cudaEvent_t start, stop;
		cudaErrCheck(cudaEventCreate(&start));
		cudaErrCheck(cudaEventCreate(&stop));

		cudaErrCheck(cudaEventRecord(start));

		float alpha = 1.f;
		float beta  = 0.f;



		transpose_weight();


		int row_start = 0;
		int column_start = 0;
		//printf("%d,%d, \n", row_start, column_start);
		//printf("here 1 \n");

		int count = 0;
		while (row_start < this->num_layers) {

			int i = row_start;
			int j = column_start;

			//printf("%d,%d,%d,%d, \n", i, j, row_start, column_start);

			//printf("here 2 \n");

			while (j >= 0 && i < this->num_layers) {
				int j_end = j + recur_batch_size;
				if (j_end > this->seq_length) j_end = this->seq_length;

				//printf("%d,%d,%d,%d, \n", i, j, row_start, column_start);
				//printf("==   here 3 \n");

				cublasErrCheck(cublasSetStream(handle, stream_i[i]));

				for (int k = j; k < j_end; k++) {
					if (i > 0) {
						cudaErrCheck(cudaStreamWaitEvent(stream_i[i], events_h[i - 1][k], 0));
						cudaErrCheck(cudaEventDestroy(events_h[i - 1][k]));
					}
				}

				int layer_index = i;
				int sequence_index = j;

				float *W = this->weight_out + layer_index * this->mem_cell_num * this->input_dim * 8;

				cublasErrCheck(cublasSgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
										   4 * this->mem_cell_num, mini_batch * (j_end - j), this->input_dim,
										   &one, W, 4 * this->mem_cell_num,
										   this->input_data + sequence_index * this->num_elements
										   + layer_index * this->num_elements * this->seq_length, this->input_dim, &one,
										   temp_input + sequence_index * 4 * this->num_elements,
										   4 * this->mem_cell_num));


				for (int k = j; k < j_end; k++) {
					cudaErrCheck(cudaEventCreate(&events_i[layer_index][k], cudaEventDisableTiming));
					cudaErrCheck(cudaEventRecord(events_i[layer_index][k], stream_i[layer_index]));
				}

				for (int k = j; k < j_end; k++) {

					// perform R * h prev

					//printf("print access : %d, %d\n", layer_index, k);

					cublasErrCheck(cublasSetStream(handle, stream_h[layer_index]));

					sequence_index = k;

					float *R = this->weight_out + layer_index * this->mem_cell_num * this->input_dim * 8
							   + this->mem_cell_num * this->input_dim * 4;

					cublasErrCheck(cublasSgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N,
											   4 * this->mem_cell_num, mini_batch, this->input_dim,
											   &one, R, 4 * this->mem_cell_num,
											   this->output_data + sequence_index * this->num_elements
											   + layer_index * this->num_elements * (this->seq_length + 1),
											   this->mem_cell_num, &one,
											   temp_output + layer_index * 4 * this->num_elements,
											   4 * this->mem_cell_num));

					// wait for the
					cudaErrCheck(cudaStreamWaitEvent(stream_h[layer_index], events_i[layer_index][k], 0));
					cudaErrCheck(cudaEventDestroy(events_i[layer_index][k]));


					// element wise operation
					dim3 blockDim;
					dim3 gridDim;


					blockDim.x = 256;
					gridDim.x = (this->num_elements + blockDim.x - 1) / blockDim.x;


					sequence_index = k;

					//const int threadsPerBlock = 256;
					//const int blocks = (this->num_elements + cdffg - 1) / threadsPerBlock;

					element_wise_operation << < gridDim, blockDim, 0, stream_h[layer_index]>> > (this->mem_cell_num, this->mini_batch,
							temp_input + sequence_index * 4 * this->num_elements,
							temp_output + layer_index * 4 * this->num_elements,
							this->bias + layer_index * mem_cell_num * 8, this->output_data +
																		 (sequence_index + 1) * this->num_elements
																		 + layer_index * this->num_elements *
																		   (this->seq_length + 1),
							this->input_data + sequence_index * this->num_elements
							+ (layer_index + 1) * this->num_elements * this->seq_length,
							this->cell_state + sequence_index * this->num_elements
							+ layer_index * this->num_elements * (this->seq_length + 1),
							this->cell_state + (sequence_index + 1) * this->num_elements
							+ layer_index * this->num_elements * (this->seq_length + 1), this->training,
							this->activations + layer_index * seq_length * 4 * num_elements +
							sequence_index * 4 * num_elements
					);
					cudaErrCheck(cudaGetLastError());
					count++;
					//printf("%d is cound\n", count);

					if (layer_index != this->num_layers - 1) {
						cudaErrCheck(cudaEventCreate(&events_h[layer_index][k], cudaEventDisableTiming));
						cudaErrCheck(cudaEventRecord(events_h[layer_index][k], stream_h[layer_index]));
					}


				}
				i++;
				j -= recur_batch_size;
			}
			if (column_start >= this->seq_length - 2) {
				row_start++;
			} else {
				column_start += recur_batch_size;
			}
		}

		cudaErrCheck(cudaEventRecord(stop));
		cudaErrCheck(cudaEventSynchronize(stop));
		cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));

		cudaErrCheck(cudaDeviceSynchronize());


		// We're done. Print some checksums
		if (checkF) {
			float* testOutputi;
			float* testOutputh;
			float* testOutputc;


			testOutputi = (float*)malloc(this->num_elements * this->seq_length * sizeof(float));
			testOutputh = (float*)malloc(this->num_elements * this->num_layers * sizeof(float));
			testOutputc = (float*)malloc(this->num_elements * this->num_layers * sizeof(float));

			cudaErrCheck(cudaMemcpy(testOutputi, this->input_data + this->num_layers * seq_length * num_elements, seq_length * num_elements * sizeof(float), cudaMemcpyDeviceToHost));
			for (int layer = 0; layer < num_layers; layer++) {
				cudaErrCheck(cudaMemcpy(testOutputh + layer * this->num_elements, this->output_data + seq_length * num_elements + layer * (seq_length + 1) * num_elements, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
				cudaErrCheck(cudaMemcpy(testOutputc + layer * this->num_elements, this->cell_state + seq_length * num_elements + layer * (seq_length + 1) * num_elements, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
			}
			double checksumi = 0.;
			double checksumh = 0.;
			double checksumc = 0.;

			for (int m = 0; m < mini_batch; m++) {
				for (int j = 0; j < seq_length; j++) {
					for (int i = 0; i < mem_cell_num; i++) {
						checksumi += testOutputi[j * num_elements + m * mem_cell_num + i];
						if (mem_cell_num <= 8) printf("i: (%d,%d): %E\n", j, i, testOutputi[j * num_elements + m * mem_cell_num + i]);
					}
				}
				for (int j = 0; j < num_layers; j++) {
					for (int i = 0; i < mem_cell_num; i++) {
						checksumh += testOutputh[j * num_elements + m * mem_cell_num + i];
						checksumc += testOutputc[j * num_elements + m * mem_cell_num + i];
					}
				}

				if (m == 0) printf("i checksum (example %d) %E\n", m, checksumi);
				if (m == 0) printf("h checksum (example %d) %E\n", m, checksumh);
				if (m == 0) printf("c checksum (example %d) %E\n", m, checksumc);
			}

			printf("i checksum %E     ", checksumi);
			printf("c checksum %E     ", checksumc);
			printf("h checksum %E\n", checksumh);

			free(testOutputi);
			free(testOutputc);
			free(testOutputh);

		}
	}


	return elapsedTime;
}


__global__ void element_wise_operation_prop1(float *celll_state) {
	/*int index = blockIdx.x * blockDim.x + threadIdx.x;

	int num_element = mem_cell_num * mini_batch;*/



}

void LSTMNetwork::backprop() {
	const int threadsPerBlock = 256;
	const int blocks = (this->num_elements + threadsPerBlock - 1) / threadsPerBlock;

	while (1) {
		int layer_index;

		int sequence_index;

		//element_wise_operation_prop1<<<blocks, threadsPerBlock>>>(this->cell_state);

		//top_diff_h;
		//top_diff_s;
	}


}

int main(int argc, char* argv[]) {
	int seqLength;
	int numLayers;
	int hiddenSize;
	int miniBatch;

	if (argc == 5) {
		seqLength = atoi(argv[1]);
		numLayers =  atoi(argv[2]);
		hiddenSize =  atoi(argv[3]);
		miniBatch =  atoi(argv[4]);
	}
	else if (argc == 1) {
		printf("Running with default settings\n");
		seqLength = 100;
		numLayers = 4;
		hiddenSize = 512;
		miniBatch = 64;
	}
	else {
		printf("Usage: ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>\n");
		return 1;
	}

	printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n", seqLength, numLayers, hiddenSize, miniBatch);

	int numRuns = 1;

	float totalTime = 0.f;

	LSTMNetwork network(4, 512, 512, 64, 100, NULL);
	for (int run = 0; run < numRuns; run++) {

		totalTime+=network.feedforward(true);
		//totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers, true);
	}

	printf("Runtime %fms\n", totalTime / numRuns);

	return time < 0;
}



