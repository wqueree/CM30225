#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

#define SIZE 4
#define PRECISION 0.1
#define THREADS 4

typedef struct MatrixLocation {
    long i;
    long j;
} MatrixLocation;

typedef struct RelaxationBatch {
    long batchLength;
    MatrixLocation* matrixLocations;
    double** mat;
    double** temp;
    bool stop;
} RelaxationBatch;

void logSquareDoubleMatrix(double** mat) {
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

void freeDoubleMatrix(double** mat) {
    free(mat[0]);
    free(mat);
}

double doubleMean(double mat[], int n) {
    double matSum = 0.0;
    for (size_t i = 0; i < n; i++) {
        matSum += mat[i];
    }
    return matSum / n;
}

long* calculateBatchLengths() {
    long n = (SIZE - 2) * (SIZE - 2);
    long* batchLengths = (long*) calloc(THREADS, sizeof(long));
    long floor = n / THREADS;
    for (size_t i = 0; i < THREADS; i++) {
        batchLengths[i] = floor;
    }
    batchLengths[0] += n % THREADS;
    return batchLengths;
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
    for (size_t k = 0; k < batch->batchLength; k++) {
        if (!updateIndex(batch->matrixLocations[k].i, batch->matrixLocations[k].j, batch->mat, batch->temp)) {
            batch->stop = false;
        }
    }
    return NULL;
}

double** doubleMatDeepCopy(double** mat) {
    double** copy = (double**) initDoubleMatrix();
    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            copy[i][j] = mat[i][j];
        }
    }
    return copy;
}

MatrixLocation** initBatchMatrixLocations(long* batchLengths) {
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) malloc(THREADS * sizeof(MatrixLocation*));
    for (size_t i = 0; i < THREADS; i++) {
        batchMatrixLocations[i] = (MatrixLocation*) malloc(batchLengths[i] * sizeof(MatrixLocation));
    }
    return batchMatrixLocations;
}

void freeBatchMatrixLocations(MatrixLocation** batchMatrixLocations) {
    for (size_t i = 0; i < THREADS; i++) {
        free(batchMatrixLocations[i]);
    }
    free(batchMatrixLocations);
}

void calculateBatchMatrixLocations(MatrixLocation** batchMatrixLocations, long* batchLengths) {
    size_t batchNumber = 0;
    size_t batchProcessed = 0;
    for (size_t i = 1; i < SIZE - 1; i++) {
        for (size_t j = 1; j < SIZE - 1; j++) {
            batchMatrixLocations[batchNumber][batchProcessed].i = i;
            batchMatrixLocations[batchNumber][batchProcessed].j = j;
            if (++batchProcessed == batchLengths[batchNumber]) {
                batchNumber++;
                batchProcessed = 0;
            }
        }
    }
}

bool relaxationStep(double** mat) {
    double** temp = doubleMatDeepCopy(mat);
    bool stop = true;
    pthread_t threads[THREADS];
    long* batchLengths = calculateBatchLengths();
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) initBatchMatrixLocations(batchLengths);
    calculateBatchMatrixLocations(batchMatrixLocations, batchLengths);
    RelaxationBatch* batches[THREADS];

    for (size_t i = 0; i < THREADS; i++) {
        batches[i] = (RelaxationBatch*) malloc(sizeof(RelaxationBatch));
        batches[i]->batchLength = batchLengths[i];
        batches[i]->matrixLocations = batchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->temp = temp;
        batches[i]->stop = true;
        assert(pthread_create(&threads[i], NULL, manageThread, (void*) batches[i]) == 0);
    }
    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (!batches[i]->stop) {
            stop = false;
        }
        free(batches[i]);
    }
    freeDoubleMatrix(temp);
    free(batchLengths);
    freeBatchMatrixLocations(batchMatrixLocations);
    return stop;
}

void relaxation(double** mat) {
    bool stopIteration = false;
    logSquareDoubleMatrix(mat);
    while (!stopIteration) {
        stopIteration = relaxationStep(mat);
        logSquareDoubleMatrix(mat);
    }
}

int main() {
    double matArray[SIZE][SIZE] = {
        {1.0, 1.0, 1.0, 1.0}, 
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };

    double** mat = initDoubleMatrix();

    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            mat[i][j] = matArray[i][j];
        }
    }
    relaxation(mat);
    freeDoubleMatrix(mat);
    return 0;
}

