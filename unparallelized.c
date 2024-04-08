// The unparallelized version.

// Start our program.

// Include the necessary libraries.
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// The number of times you'd like to multiply
// the matrices together.
const int TRIALS = 1;

// Instead of using N for the size of the arrays, we used
// the constants ROWS and COLUMNS, below.
//const int N = 100;

// The number of rows and columns in the matrices.
const int ROWS = 100;
const int COLUMNS = 100;

// Choose a number to fill the first matrix.
const int MATRIX_1_NUMBER = 7;
// Choose a number to fill the second matrix.
const int MATRIX_2_NUMBER = 8;

int main() {

    int trial, i;

    // These will be used to quantify the amount of time
    // the program took to run.
    struct timespec start, stop;

    // Allocate and initialize memory for the matrices.
    int **matrix1, **matrix2, **matrix3;

    // Allocate and initialize rows memory.
    matrix1 = (int**)malloc(ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix1[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix2 = (int**)malloc(ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix2[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix3 = (int**)malloc(ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < ROWS; i++) {
        matrix3[i] = (int *)malloc(COLUMNS * sizeof(int));
    }

    // Fill matrix1 with an integer.
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix1[i][j] = MATRIX_1_NUMBER;
        }
    }

    // Fill matrix2 with an integer.
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix2[i][j] = MATRIX_2_NUMBER;
        }
    }

    // Record the time the computation started.
    clock_gettime(CLOCK_REALTIME, &start);

    // Perform matrix multiplication.
    for (trial = 0; trial < TRIALS; trial++) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                for(int x = 0; x < COLUMNS; x++) {
                    // Compute the result.
                    // Store the result in matrix3.
                    matrix3[i][j] += matrix1[i][x] * matrix2[x][j];
                }
            }
        }
    };

    // Record the time the computation finished.
    clock_gettime(CLOCK_REALTIME, &stop);

    // Record execution time as the difference of event timestamps.
    double milliseconds = (stop.tv_sec - start.tv_sec) * 1e3 +
    (stop.tv_nsec - start.tv_nsec) / 1e6;

    /*
    // This part is commented out because it becomes impractical for large matrices.

    // Print the result to check its accuracy.
    for (i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            // Print the matrix element.
            printf("%d ", matrix3[i][j]);
        }
        // Go to the next line for the next row.
        printf("\n");
    }
    */

    // Measure the performance by printing the duration.
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
