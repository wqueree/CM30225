#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define BILLION 1000000000L // Used in time calculations

#define LOGGING true // Switch for logging
#define PRECISION 0.01 // Sets the precision


typedef struct MatrixLocation {
    // Defines indices for a location in a matrix
    long i;
    long j;
} MatrixLocation;

typedef struct ReadBatch {
    // Defines data required to read batch from file in parallel
    size_t batchLength;
    MatrixLocation* matrixLocations;
    double** mat;
    double** copy;
} ReadBatch;

typedef struct WriteBatch {
    // Defines data required to calculate and write data to mat in memory in parallel
    size_t batchLength;
    MatrixLocation* matrixLocations;
    double** mat;
    double** copy;
    bool stop;
} WriteBatch;

double** initDoubleMatrix(size_t size) {
    // Allocates memory for a double matrix of size*size elements
    double** mat = (double**) malloc(size * sizeof(double*));
    double* matBuf = (double*) malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

void freeDoubleMatrix(double** mat) {
    // Frees memory allocated by initDoubleMatrix
    free(mat[0]);
    free(mat);
}

void logSquareDoubleMatrix(double** mat, size_t size) {
    // Logs mat
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.2lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double doubleMean(double values[], int n) {
    // Calculates mean of the n elements of values
    double valuesSum = 0.0;
    for (int i = 0; i < n; i++) {
        valuesSum += values[i];
    }
    return valuesSum / n;
}

void timespecDifference(struct timespec start, struct timespec stop, struct timespec* delta) {
    // Calculates the difference between two struct timespecs
    delta->tv_nsec = stop.tv_nsec - start.tv_nsec;
    delta->tv_sec  = stop.tv_sec - start.tv_sec;
    if (delta->tv_sec > 0 && delta->tv_nsec < 0) {
        delta->tv_nsec += BILLION;
        delta->tv_sec--;
    }
    else if (delta->tv_sec < 0 && delta->tv_nsec > 0) {
        delta->tv_nsec -= BILLION;
        delta->tv_sec++;
    }
}

void logDuration(size_t size, double duration, size_t threads) {
    // Logs a double duration with some additional semantic info
    printf("%ldx%ld matrix converged at precision %lf in %lfs using %ld threads\n", size, size, PRECISION, duration, threads);
}

double doubleTime(struct timespec delta) {
    // Converts timespec time to double time
    return (double) delta.tv_sec + ((double) delta.tv_nsec / BILLION);
}