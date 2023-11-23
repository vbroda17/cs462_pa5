

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void calculateMean(int* data, float* mean) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute mean
    atomicAdd(mean, data[tid]);

    // Print information
    //printf("Thread %d: Adding %f to mean\n", tid, data[tid]);
}

__global__ void calculateVariance(int* data, float mean, float* variance) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("The mean for variance is %f\n", mean);
    if (tid < N) {
        // Synchronize threads to ensure accurate mean calculation

        // Compute variance
        float diff = data[tid] - mean;
        atomicAdd(variance, diff * diff);

        // Print information
        //printf("Thread %d: Adding %f to variance\n", tid, diff * diff);
    }
}

int main() {
    // Allocate memory on host
    int* h_data = (int*)malloc(N * sizeof(int));
    float* h_mean = (float*)malloc(sizeof(float));
    float* h_variance = (float*)malloc(sizeof(float));

    // Initialize the array with values 1 to N, in this case 1,000,000
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;
        //printf("%d ", h_data[i]);
    }
    //printf("\n");
    int* d_data;
    float * d_mean, * d_variance;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMalloc((void**)&d_mean, sizeof(float));
    cudaMalloc((void**)&d_variance, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Set initial values for mean and variance on host
    *h_mean = 0.0f;
    *h_variance = 0.0f;

    // Copy initial values from host to device
    cudaMemcpy(d_mean, h_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to calculate mean
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    calculateMean <<<gridSize, blockSize >>> (d_data, d_mean);

    // Synchronize and copy mean result from device to host
    cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final mean
    *h_mean /= N;

    // Launch kernel to calculate variance
    calculateVariance << <gridSize, blockSize >> > (d_data, *h_mean, d_variance);

    // Synchronize and copy variance result from device to host
    cudaMemcpy(h_variance, d_variance, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final variance, specifically the sample variance, if we want population variance we get rid of the -1
    *h_variance /= (N - 1);

    // Display results
    printf("\nMean: %f\n", *h_mean);
    printf("Variance: %f\n", *h_variance);

    // Free allocated memory
    free(h_data);
    free(h_mean);
    free(h_variance);
    cudaFree(d_data);
    cudaFree(d_mean);
    cudaFree(d_variance);

    return 0;
}
