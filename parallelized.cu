#include <stdio.h>
#include <cuda_runtime.h>

// This is Kernel function that uses the GPU to perform matrix multiplication
__global__ void matrixMulGlobalKernel(int* MatrixA, int* MatrixB, int* MatrixC, int m, int n, int k) {
    // calc the row index of the output matrix element allocated by the current thread
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    // Calc the column index of the output matrix element allocated by the current thread
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    //Init the accumulator variable of the current output element
    int accumulator = 0;

    //we need to make sure current thread is within the valid range of the output matrix
    if (nRow < m && nCol < n) {
        // Traverse corresponding MatrixA and MatrixB, and calculate the dot product
        for (int i = 0; i < k; i++) {
            // Accumulate the product of the corresponding elements of MatrixA and MatrixB
            accumulator += MatrixA[nRow * k + i] * MatrixB[i * n + nCol];
        }
        // Store the calc results in the corresponding elements of the output MatrixC
        MatrixC[nRow * n + nCol] = accumulator;
    }
}

int main() {
    const int M = 3;//Number of rows of output MatrixC
    const int N = 3;//Number of cols of output MatrixC
    const int K = 3;//intermediate dimension = matrix A col = matrix B row

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
    dim3 blockDim(3, 3);//we need change block size (9, 9) ,(81, 81), (243, 243);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Create CUDA event
    cudaEvent_t start;
    cudaEvent_t stop;
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
