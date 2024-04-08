#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMulGlobalKernel(int* pfMatrixA, int* pfMatrixB, int* pfMatrixC, int m, int n, int k) {
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    int fCVal = 0;

    if (nRow < m && nCol < n) {
        for (int i = 0; i < k; i++) {
            fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
        }

        pfMatrixC[nRow * n + nCol] = fCVal;
    }
}

int main() {
    // Number of rows in Matrix 1 and Matrix 3.
    const int M = 1000;
    // Number of columns in Matrix 2 and Matrix 3.
    const int N = 1000;
    // Number of columns in Matrix 1 and rows in Matrix 2.
    const int K = 1000;

    int* h_A = (int*)malloc(M * K * sizeof(int));
    int* h_B = (int*)malloc(K * N * sizeof(int));
    int* h_C = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 7;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 8;
    }

    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * K * sizeof(int));
    cudaMalloc(&d_B, K * N * sizeof(int));
    cudaMalloc(&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulGlobalKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    /*
    printf("Result Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }
    */

    printf("Global Memory Kernel Elapsed Time: %f milliseconds\n", milliseconds);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
