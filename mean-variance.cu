// Vincent Broda
// CS462 Assignment 5, mean and variance calculations with cuda. See read me for more specific explinations of code and ideas.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

// using atomic add to calculate the summation of all the data in the set
__global__ void calculateMean(int* data, float* mean) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(mean, data[tid]);
}

// doing something simlar, this time the summation that we need is slightly different though
__global__ void calculateVariance(int* data, float mean, float* variance) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // if (tid < N) {     // Originally had this if statement to try to do some type of check, but I think it was just becasuse of my paranorma and i removed it. I did this after my test on the isaac node though 
    float diff = data[tid] - mean;
    atomicAdd(variance, diff * diff);
}

int main() {
    // Allocate memory on host
    int* h_data = (int*)malloc(N * sizeof(int));
    float* h_mean = (float*)malloc(sizeof(float));
    float* h_variance = (float*)malloc(sizeof(float));

    // Memory for the device
    int* d_data;
    float * d_mean, * d_variance;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMalloc((void**)&d_mean, sizeof(float));
    cudaMalloc((void**)&d_variance, sizeof(float));

    // Initialize the array with values 1 to N, in this case 1,000,000
    for (int i = 0; i < N; i++) h_data[i] = i + 1;

    // Copy data set from host to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    *h_mean = 0.0f;
    *h_variance = 0.0f;

    // Copy initial values from host to device, I dont think this is actually needed but all well, I'm sure it hurts preformance a little but it is not noticable
    cudaMemcpy(d_mean, h_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to calculate mean, we will alco be using these block sizes, I'm not sure if theres a better way to decide this or not, but again the preformance is good enough for it to be fine I belive 
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    calculateMean<<<gridSize, blockSize>>>(d_data, d_mean);

    // Synchronize result from device to host
    cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final mean, we will do division here for efficiency
    *h_mean /= N;

    // Launch kernel to calculate variance, very similar to mean, but just a bit more complicated becasue it is the variance
    calculateVariance<<<gridSize, blockSize>>>(d_data, *h_mean, d_variance);

    // Synchronize result from device to host
    cudaMemcpy(h_variance, d_variance, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final variance, specifically the sample variance, if we want population variance we get rid of the -1
    *h_variance /= (N - 1);

    printf("Mean: %f\n", *h_mean);
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
