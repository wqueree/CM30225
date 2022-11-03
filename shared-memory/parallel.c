#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

#define SIZE 4
#define PRECISION 0.001
#define THREADS 1

typedef struct MatrixLocation {
    long i;
    long j;
} MatrixLocation;

typedef struct RelaxationBatch {
    long batchLength;
    double** start;
    double** batchStart;
    double** array;
    double** temp;
    bool stop;
} RelaxationBatch;

void logSquareDoubleArray(double** mat) {
    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** initDoubleMatrix() {
    double** mat = (double**) malloc(SIZE * sizeof(double*));
    double* matBuf = malloc(SIZE * SIZE * sizeof(double));
    for (size_t i = 0; i < SIZE; i++) {
        mat[i] = (SIZE * i) + matBuf;
    }
    return mat;
}

double doubleMean(double array[], int n) {
    double arraySum = 0.0;
    for (size_t i = 0; i < n; i++) {
        arraySum += array[i];
    }
    return arraySum / n;
}

long* calculateThreadLengths() {
    long n = (SIZE - 2) * (SIZE - 2);
    long* threadLengths = (long*) calloc(THREADS, sizeof(long));
    long floor = n / THREADS;
    for (size_t i = 0; i < THREADS; i++) {
        threadLengths[i] = floor;
    }
    threadLengths[0] += n % THREADS;
    return threadLengths;
}

bool updateIndex(long i, long j, double** mat, double** temp) {
    double meanValues[SIZE] = {
        temp[i - 1][j],
        temp[i][j + 1],
        temp[i + 1][j],
        temp[i][j - 1],
    };
    mat[i][j] = doubleMean(meanValues, 4);
    bool stop = fabs(mat[i][j] - temp[i][j]) < PRECISION;
    return stop;
}

MatrixLocation* calculateMatrixLocation(double** start, double** current) {
    long indicesFromStart = current - start;
    MatrixLocation* matrixLocation = (MatrixLocation*) malloc(sizeof(MatrixLocation));
    matrixLocation->i = indicesFromStart / SIZE;
    matrixLocation->j = indicesFromStart % SIZE;
    return matrixLocation;
}

void* manageThread(void* voidBatch) {
    RelaxationBatch* batch = (RelaxationBatch*) voidBatch;
    double** current = batch->batchStart;
    for (size_t i = 0; i < batch->batchLength; i++) {
        MatrixLocation* matrixLocation = calculateMatrixLocation(batch->start, current);
        if (!updateIndex(matrixLocation->i, matrixLocation->j, batch->array, batch->temp)) {
            batch->stop = false;
        }
        free(matrixLocation);
    }
    return NULL;
}

bool relaxationStep(double** array) {
    double** temp = (double**) initDoubleMatrix();
    memcpy(temp, array, sizeof(double*) * SIZE * SIZE); // TODO Copy mat values not pointers
    double** start = temp;
    bool stopIteration = true;
    pthread_t threads[THREADS];
    long* batchLengths = calculateThreadLengths();
    long processed = 0;
    RelaxationBatch* batches[THREADS];
    for (size_t i = 0; i < THREADS; i++) {
        batches[i] = (RelaxationBatch*) malloc(sizeof(RelaxationBatch));
        batches[i]->batchLength = batchLengths[i];
        batches[i]->start = start;
        batches[i]->batchStart = (double**) ((long) start + ((SIZE + 1 + processed) * sizeof(double*)));
        batches[i]->array = array;
        batches[i]->temp = temp;
        batches[i]->stop = true;
        assert(pthread_create(&threads[i], NULL, manageThread, (void*) batches[i]) == 0);
        processed += batchLengths[i];
    }
    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (!batches[i]->stop) {
            stopIteration = false;
            free(&batches[i]);
        }
    }
    return stopIteration;
}

void relaxation(double** mat) {
    bool stopIteration = false;
    logSquareDoubleArray(mat);
    while (!stopIteration) {
        stopIteration = relaxationStep(mat);
        logSquareDoubleArray(mat);
    }
}

int main() {
    double secondOrder[SIZE][SIZE] = {
        {1.0, 1.0, 1.0, 1.0}, 
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };

    double** secondOrderPtr = (double**) initDoubleMatrix();

    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            secondOrderPtr[i][j] = secondOrder[i][j];
        }
    }

    relaxation(secondOrderPtr);
    return 0;
}
