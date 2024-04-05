// The unparallelized version.

// Start our program.

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const int TRIALS = 1;
const int N = 100;

const int ROWS = 3;
const int COLUMNS = 3;

int main() {

    int trial, i;
    struct timespec start, stop;

    // Allocate and initialize memory.
    int **matrix1, **matrix2, **matrix3;

    // Allocate and initialize rows memory.
    matrix1 = (int**)malloc(ROWS * sizeof(int));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix1[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix2 = (int*)malloc(N * sizeof(int));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix2[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix3 = (int*)malloc(N * sizeof(int));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix3[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Fill matrix1 with the number 7.
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix1[i][j] = 7;
        }
    }

    // Fill matrix2 with the number 8.
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix2[i][j] = 8;
        }
    }

    clock_gettime(CLOCK_REALTIME, &start);

    // Perform matrix multiplication.
    for (trial = 0; trial < TRIALS; trial++) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                // Code the computation here.
                // Store the result in matrix3.
            }
        }
    };

    clock_gettime(CLOCK_REALTIME, &stop);

    // record execution time as the difference of event timestamps
    double milliseconds = (stop.tv_sec - start.tv_sec) * 1e3 +
    (stop.tv_nsec - start.tv_nsec) / 1e6;

    for (i = 0; i < N; i++) printf("%d ", c[i]);
    printf("\n");
    printf("Serial duration: %3.6fms\n", milliseconds);
}

// End our program.

/****************************/

// Below is the code from the professor's "serial-add-timed.c" file.
// Use it as a reference when writing our own code.

/*

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const int TRIALS = 1;
const int N = 100;

int main() {
  int trial, i;
  struct timespec start, stop;

  // allocate and initialize memory
  int *a, *b, *c;
  a = (int*)malloc(N * sizeof(int));
  b = (int*)malloc(N * sizeof(int));
  c = (int*)malloc(N * sizeof(int));
  for (i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  clock_gettime(CLOCK_REALTIME, &start);

  // perform the computation
  for (trial = 0; trial < TRIALS; trial++) {
    for (i = 0; i < N; i++) {
      c[i] += a[i] + b[i];
    }
  };

  clock_gettime(CLOCK_REALTIME, &stop);

  // record execution time as the difference of event timestamps
  double milliseconds = (stop.tv_sec - start.tv_sec) * 1e3 +
    (stop.tv_nsec - start.tv_nsec) / 1e6;

  for (i = 0; i < N; i++) printf("%d ", c[i]);
  printf("\n");
  printf("Serial duration: %3.6fms\n", milliseconds);
}

*/