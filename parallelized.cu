#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMulGlobalKernel(int* MatrixA, int* MatrixB, int* MatrixC, int m, int n, int k) {
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    int fCVal = 0;

    if (nRow < m && nCol < n) {
        for (int i = 0; i < k; i++) {
            fCVal += MatrixA[nRow * k + i] * MatrixB[i * n + nCol];
        }

        MatrixC[nRow * n + nCol] = fCVal;
    }
}

int main() {
    const int M = 3;
    const int N = 3;
    const int K = 3;

    //Allocate the matrix on the host;
    int* h_A = (int*)malloc(M * K * sizeof(int));
    int* h_B = (int*)malloc(K * N * sizeof(int));
    int* h_C = (int*)malloc(M * N * sizeof(int));

    //Initialize the matrix on the host;
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 7;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 8;
    }

    // Allocate the matrix on the device
    int* d_A;
    int * d_B;
    int * d_C;
    cudaMalloc(&d_A, M * K * sizeof(int));
    cudaMalloc(&d_B, K * N * sizeof(int));
    cudaMalloc(&d_C, M * N * sizeof(int));

    // Copy data from host memory to device memory
    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Create CUDA event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Call CUDA kernel function
    matrixMulGlobalKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate running time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results from device memory to host memory
    cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Release allocated memory
    printf("Global Memory Kernel Elapsed Time: %f milliseconds\n", milliseconds);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
