#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define BILLION 1000000000L

#define LOGGING false
#define PRECISION 0.01

typedef struct MatrixLocation {
    long i;
    long j;
} MatrixLocation;

typedef struct ReadBatch {
    size_t batchLength;
    MatrixLocation* matrixLocations;
    double** mat;
    double** copy;
} ReadBatch;

typedef struct WriteBatch {
    size_t batchLength;
    MatrixLocation* matrixLocations;
    double** mat;
    double** copy;
    bool stop;
} WriteBatch;

double** initDoubleMatrix(size_t size) {
    double** mat = (double**) malloc(size * sizeof(double*));
    double* matBuf = (double*) malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

void freeDoubleMatrix(double** mat) {
    free(mat[0]);
    free(mat);
}

void logSquareDoubleMatrix(double** mat, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.2lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

pthread_mutex_t** initMutexMatrix(size_t size) {
    pthread_mutex_t** mtxMat = (pthread_mutex_t**) malloc(size * sizeof(pthread_mutex_t*));
    pthread_mutex_t* mtxMatBuf = (pthread_mutex_t*) malloc(size * size * sizeof(pthread_mutex_t));
    for (size_t i = 0; i < size; i++) {
        mtxMat[i] = (size * i) + mtxMatBuf;
    }
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            pthread_mutex_init(&mtxMat[i][j], NULL);
        }
    }
    return mtxMat;
}

void freeMutexMatrix(pthread_mutex_t** mtxMat) {
    free(mtxMat[0]);
    free(mtxMat);
}

double doubleMean(double mat[], int n) {
    double matSum = 0.0;
    for (int i = 0; i < n; i++) {
        matSum += mat[i];
    }
    return matSum / n;
}

void timespecDifference(struct timespec start, struct timespec stop, struct timespec* delta) {
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
    printf("%ldx%ld matrix converged at precision %lf in %lfs using %ld threads\n", size, size, PRECISION, duration, threads);
}

double doubleTime(struct timespec delta) {
    return (double) delta.tv_sec + ((double) delta.tv_nsec / BILLION);
}