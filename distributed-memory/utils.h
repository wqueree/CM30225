#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define BILLION 1000000000L // Used in time calculations

#define LOGGING false // Switch for logging
#define PRECISION 0.01 // Sets the precision

typedef struct FlatMatrixChunk {
    size_t n;
    size_t m;
    size_t start_row;
    double* flat;
} FlatMatrixChunk;

double** initSquareDoubleMatrix(size_t size) {
    // Allocates memory for a double matrix of size*size elements
    double** mat = (double**) malloc(size * sizeof(double*));
    double* matBuf = (double*) malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

double** inputDoubleMatrix(char* dataFilePath, size_t* size) {
    FILE* dataFile = fopen(dataFilePath, "r");
    fscanf(dataFile, "%ld", size);
    double** mat = initSquareDoubleMatrix(*size);
    for (size_t i = 0; i < *size; i++) {
        for (size_t j = 0; j < *size; j++) {
            fscanf(dataFile, "%lf", &mat[i][j]);
        }
    }
    fclose(dataFile);
    return mat;
}

double** initDoubleMatrix(size_t n, size_t m) {
    // Allocates memory for a double matrix of size*size elements
    double** mat = (double**) malloc(n * sizeof(double*));
    double* matBuf = (double*) malloc(n * m * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        mat[i] = (m * i) + matBuf;
    }
    return mat;
}

void doubleMatrixDeepCopy(double** mat, double** cpy, size_t n, size_t m) {
    // Creates copy of double matrix serially.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            cpy[i][j] = mat[i][j];
        }
    }
}

void squareDoubleMatrixDeepCopy(double** mat, double** cpy, size_t size) {
    // Creates copy of square double matrix serially.
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            cpy[i][j] = mat[i][j];
        }
    }
}

double** reshapeRows(double* flat, size_t n, size_t m) {
    double** mat = initDoubleMatrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            mat[i][j] = flat[(i * m) + j];
        }
    }
    return mat;
}


void freeDoubleMatrix(double** mat) {
    // Frees memory allocated by initDoubleMatrix
    free(mat[0]);
    free(mat);
}

void logDoubleMatrix(double** mat, size_t n, size_t m) {
    // Logs mat
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%.2lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
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

void matrixSwap(double*** mat, double*** cpy) {
    double** tmp = *mat;
    *mat = *cpy;
    *cpy = tmp;
}

double doubleMean(double values[], int n) {
    // Calculates mean of the n elements of values
    double valuesSum = 0.0;
    for (int i = 0; i < n; i++) {
        valuesSum += values[i];
    }
    return valuesSum / n;
}

double calculateNeighbourMean(double** mat, size_t i, size_t j) {
    double neighbours[] = {
        mat[i - 1][j],
        mat[i][j + 1],
        mat[i + 1][j],
        mat[i][j - 1]
    };
    return doubleMean(neighbours, 4);
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

void logDuration(size_t size, double duration, size_t n_processors) {
    // Logs a double duration with some additional semantic info
    printf("%ldx%ld matrix converged at precision %lf in %lfs using %ld processors\n", size, size, PRECISION, duration, n_processors);
}

double doubleTime(struct timespec delta) {
    // Converts timespec time to double time
    return (double) delta.tv_sec + ((double) delta.tv_nsec / BILLION);
}