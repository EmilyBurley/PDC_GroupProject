// The parallelized version.

// Start our program.



// End our program.

/****************************/

// Below is the code from the professor's "parallel-add-timed.cu" file.
// Use it as a reference when writing our own code.

/*

#include <stdlib.h>
#include <stdio.h>

const int TRIALS = 1;
const int N = 100;

__global__ void kernel_add(int* a, int* b, int* c) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (int trial = 0; trial < TRIALS; trial++) {
    c[tid] += a[tid] + b[tid];
  }
}

int main() {
  // allocate events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate and initialize host memory
  int *a, *b, *c;
  a = (int*)malloc(N * sizeof(int));
  b = (int*)malloc(N * sizeof(int));
  c = (int*)malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // allocate device memory
  int *dev_a, *dev_b, *dev_c;
  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));

  // copy data from host memory to device memory
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // record a timestamp before the kernel begins execution
  cudaEventRecord(start);

  // perform a computation on the device
  kernel_add<<<1, N>>>(dev_a, dev_b, dev_c);

  // record a timestamp after the kernel completes
  cudaEventRecord(stop);

  // copy results from the device to the host
  cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  // block until the stop event is recorded
  cudaEventSynchronize(stop);

  // record execution time as the difference of event timestamps
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  for (int i = 0; i < N; i++) printf("%d ", c[i]);
  printf("\n");
  printf("Kernel duration: %3.6fms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

*/