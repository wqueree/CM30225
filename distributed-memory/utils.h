#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define BILLION 1000000000L // Used in time calculations

#define LOGGING false // Switch for logging
#define PRECISION 0.01 // Sets the precision

float** initSquareFloatMatrix(size_t size) {
    // Allocates memory for a float matrix of size*size elements
    float** mat = (float**) malloc(size * sizeof(float*));
    float* matBuf = (float*) malloc(size * size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

float** inputFloatMatrix(char* dataFilePath, size_t* size) {
    FILE* dataFile = fopen(dataFilePath, "r");
    fscanf(dataFile, "%ld", size);
    float** mat = initSquareFloatMatrix(*size);
    for (size_t i = 0; i < *size; i++) {
        for (size_t j = 0; j < *size; j++) {
            fscanf(dataFile, "%f", &mat[i][j]);
        }
    }
    fclose(dataFile);
    return mat;
}

float** initFloatMatrix(size_t n, size_t m) {
    // Allocates memory for a float matrix of size*size elements
    float** mat = (float**) malloc(n * sizeof(float*));
    float* matBuf = (float*) malloc(n * m * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        mat[i] = (m * i) + matBuf;
    }
    return mat;
}

void floatMatrixDeepCopy(float** mat, float** cpy, size_t n, size_t m) {
    // Creates copy of float matrix serially.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            cpy[i][j] = mat[i][j];
        }
    }
}

void squareFloatMatrixDeepCopy(float** mat, float** cpy, size_t size) {
    // Creates copy of square float matrix serially.
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            cpy[i][j] = mat[i][j];
        }
    }
}

void freeFloatMatrix(float** mat) {
    // Frees memory allocated by initFloatMatrix
    free(mat[0]);
    free(mat);
}

void logFloatMatrix(float** mat, size_t n, size_t m) {
    // Logs mat
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void logSquareFloatMatrix(float** mat, size_t size) {
    // Logs mat
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void logFloatArray(float* arr, size_t size) {
    for (size_t i = 0 ; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

float floatMean(float values[], int n) {
    // Calculates mean of the n elements of values
    float valuesSum = 0.0;
    for (int i = 0; i < n; i++) {
        valuesSum += values[i];
    }
    return valuesSum / (float) n;
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

void logDuration(size_t size, float duration, size_t nProcessors) {
    // Logs a float duration with some additional semantic info
    printf("%ldx%ld matrix converged at precision %lf in %lfs using %ld processors\n", size, size, PRECISION, duration, nProcessors);
}

float floatTime(struct timespec delta) {
    // Converts timespec time to float time
    return (float) delta.tv_sec + ((float) delta.tv_nsec / BILLION);
}