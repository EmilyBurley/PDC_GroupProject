// The unparallelized version.

// Start our program.

// Include the necessary libraries.
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// The number of times you'd like to multiply the two matrices
// together and add the result to matrix3.
const int TRIALS = 1;

// The number of rows in matrix1.
const int M1_ROWS = 3;

// The number of columns in matrix1, and the number of rows in
// matrix2. By the constraints of matrix multiplication, this
// number must be the same.
const int M1_COLUMNS_M2_ROWS = 4;

// The number of columns in matrix2.
const int M2_COLUMNS = 5;

// The number of rows in matrix3, which is equals the number of
// rows in matrix1.
const int M3_ROWS = M1_ROWS;

// The number of columns in matrix3, which is equals the number of
// columns in matrix2.
const int M3_COLUMNS = M2_COLUMNS;

// Choose a number to fill the first matrix.
const int MATRIX_1_NUMBER = 7;
// Choose a number to fill the second matrix.
const int MATRIX_2_NUMBER = 8;

int main() {

    // These will be used to quantify the amount of time
    // the program took to run.
    struct timespec start, stop;

    // Allocate and initialize memory for the matrices.
    int **matrix1, **matrix2, **matrix3;

    // Allocate and initialize rows memory.
    matrix1 = (int**)malloc(M1_ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < M1_ROWS; i++) {
        matrix1[i] = (int *)malloc(M1_COLUMNS_M2_ROWS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix2 = (int**)malloc(M1_COLUMNS_M2_ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < M1_COLUMNS_M2_ROWS; i++) {
        matrix2[i] = (int *)malloc(M2_COLUMNS * sizeof(int));
    }

    // Allocate and initialize rows memory.
    matrix3 = (int**)malloc(M3_ROWS * sizeof(int*));
    // Put an array in each row.
    for (int i = 0; i < M3_ROWS; i++) {
        matrix3[i] = (int *)malloc(M3_COLUMNS * sizeof(int));
    }

    // Fill matrix1 with an integer.
    for (int i = 0; i < M1_ROWS; i++) {
        for (int j = 0; j < M1_COLUMNS_M2_ROWS; j++) {
            matrix1[i][j] = MATRIX_1_NUMBER;
        }
    }

    // Fill matrix2 with an integer.
    // For each row...
    for (int i = 0; i < M1_COLUMNS_M2_ROWS; i++) {
        // ... fill each element in that row.
        for (int j = 0; j < M2_COLUMNS; j++) {
            matrix2[i][j] = MATRIX_2_NUMBER;
        }
    }

    // Record the time the computation started.
    clock_gettime(CLOCK_REALTIME, &start);

    // Perform the computation of matrix multiplication.
    // In each trial,
    for (int trial = 0; trial < TRIALS; trial++) {
        // ... for each row of matrix3...
        for (int row = 0; row < M3_ROWS; row++) {
            // ... and for each column within that row...
            for (int column = 0; column < M3_COLUMNS; column++) {
                // ... iterate through matrix1's row and matrix2's column...
                for(int x = 0; x < M1_COLUMNS_M2_ROWS; x++) {
                    // ... multiplying that element of each matrix together...
                    // ... and adding the result to the current location in matrix3.
                    matrix3[row][column] += matrix1[row][x] * matrix2[x][column];
                }
            }
        }
    };

    // Record the time the computation finished.
    clock_gettime(CLOCK_REALTIME, &stop);

    // Record execution time as the difference of event timestamps.
    double milliseconds = (stop.tv_sec - start.tv_sec) * 1e3 +
    (stop.tv_nsec - start.tv_nsec) / 1e6;


    // This part is commented out because it becomes impractical for large matrices.
    /*
    // Print the result to check its accuracy.
    for (int i = 0; i < M3_ROWS; i++) {
        for (int j = 0; j < M3_COLUMNS; j++) {
            // Print the matrix element.
            printf("%d ", matrix3[i][j]);
        }
        // Go to the next line for the next row.
        printf("\n");
    }
    */

    // Measure the performance by printing the duration.
    printf("Serial/unparallelized program duration: %3.6fms\n", milliseconds);
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
