#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

#define PRECISION 0.01
#define THREADS 4

typedef struct MatrixLocation {
    long i;
    long j;
} MatrixLocation;

typedef struct RelaxationBatch {
    size_t batchLength;
    MatrixLocation* matrixLocations;
    pthread_mutex_t mat_mtx;
    double** mat;
    double** temp;
    bool stop;
} RelaxationBatch;

void logSquareDoubleMatrix(double** mat, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%.2lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** initDoubleMatrix(size_t size) {
    double** mat = (double**) malloc(size * sizeof(double*));
    double* matBuf = malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

void freeDoubleMatrix(double** mat) {
    free(mat[0]);
    free(mat);
}

double doubleMean(double mat[], size_t n) {
    double matSum = 0.0;
    for (size_t i = 0; i < n; i++) {
        matSum += mat[i];
    }
    return matSum / (double) n;
}

long* calculateBatchLengths(size_t size) {
    long n = (long) ((size - 2) * (size - 2));
    long* batchLengths = (long*) calloc(THREADS, sizeof(long));
    long floor = n / THREADS;
    for (size_t i = 0; i < THREADS; i++) {
        batchLengths[i] = floor;
    }
    batchLengths[0] += n % THREADS;
    return batchLengths;
}

bool updateIndex(long i, long j, double** mat, double** temp, pthread_mutex_t mat_mtx) {
    double meanValues[] = {
        temp[i - 1][j],
        temp[i][j + 1],
        temp[i + 1][j],
        temp[i][j - 1],
    };
    
    pthread_mutex_lock(&mat_mtx);
    mat[i][j] = doubleMean(meanValues, 4);
    pthread_mutex_unlock(&mat_mtx);

    bool stop = fabs(mat[i][j] - temp[i][j]) < PRECISION;
    return stop;
}

void* manageThread(void* voidBatch) {
    RelaxationBatch* batch = (RelaxationBatch*) voidBatch;
    for (size_t k = 0; k < batch->batchLength; k++) {
        if (!updateIndex(batch->matrixLocations[k].i, batch->matrixLocations[k].j, batch->mat, batch->temp, batch->mat_mtx)) {
            batch->stop = false;
        }
    }
    return NULL;
}

double** doubleMatrixDeepCopy(double** mat, size_t size) {
    double** copy = (double**) initDoubleMatrix(size);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            copy[i][j] = mat[i][j];
        }
    }
    return copy;
}

MatrixLocation** initBatchMatrixLocations(long* batchLengths) {
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) malloc(THREADS * sizeof(MatrixLocation*));
    for (size_t i = 0; i < THREADS; i++) {
        batchMatrixLocations[i] = (MatrixLocation*) malloc((size_t) batchLengths[i] * sizeof(MatrixLocation));
    }
    return batchMatrixLocations;
}

void freeBatchMatrixLocations(MatrixLocation** batchMatrixLocations) {
    for (size_t i = 0; i < THREADS; i++) {
        free(batchMatrixLocations[i]);
    }
    free(batchMatrixLocations);
}

void calculateBatchMatrixLocations(MatrixLocation** batchMatrixLocations, long* batchLengths, size_t size) {
    size_t batchNumber = 0;
    size_t batchProcessed = 0;
    for (long i = 1; i < (long) size - 1; i++) {
        for (long j = 1; j < (long) size - 1; j++) {
            batchMatrixLocations[batchNumber][batchProcessed].i = i;
            batchMatrixLocations[batchNumber][batchProcessed].j = j;
            if (++batchProcessed == (size_t) batchLengths[batchNumber]) {
                batchNumber++;
                batchProcessed = 0;
            }
        }
    }
}

bool relaxationStep(double** mat, size_t size, pthread_mutex_t mat_mtx) {
    double** temp = doubleMatrixDeepCopy(mat, size);
    bool stop = true;
    pthread_t threads[THREADS];
    long* batchLengths = calculateBatchLengths(size);
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) initBatchMatrixLocations(batchLengths);
    calculateBatchMatrixLocations(batchMatrixLocations, batchLengths, size);
    RelaxationBatch* batches[THREADS];

    for (size_t i = 0; i < THREADS; i++) {
        batches[i] = (RelaxationBatch*) malloc(sizeof(RelaxationBatch));
        batches[i]->batchLength = (size_t) batchLengths[i];
        batches[i]->matrixLocations = batchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->mat_mtx = mat_mtx;
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

void relaxation(double** mat, size_t size, pthread_mutex_t mat_mtx) {
    bool stop = false;
    logSquareDoubleMatrix(mat, size);
    while (!stop) {
        stop = relaxationStep(mat, size, mat_mtx);
        logSquareDoubleMatrix(mat, size);
    }
}

int main(int argc, char** argv) {
    char* dataFilePath = argv[1];
    FILE* dataFile = fopen(dataFilePath, "r");

    size_t size = 0;

    fscanf(dataFile, "%ld", &size);

    double** mat = initDoubleMatrix(size);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            fscanf(dataFile, "%lf", &mat[i][j]);
        }
    }

    fclose(dataFile);

    pthread_mutex_t mat_mtx;
    pthread_mutex_init(&mat_mtx, NULL);

    relaxation(mat, size, mat_mtx);
    freeDoubleMatrix(mat);
    return 0;
}
