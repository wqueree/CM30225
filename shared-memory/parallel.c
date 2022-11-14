#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

void calculateReadBatchLengths(long* readBatchLengths, size_t size, size_t n_threads) {
    long n = (long) (size * size);
    long floor = n / (long) n_threads;
    for (size_t i = 0; i < n_threads; i++) {
        readBatchLengths[i] = floor;
    }
    readBatchLengths[0] += n % (long) n_threads;
}

void calculateWriteBatchLengths(long* writeBatchLengths, size_t size, size_t n_threads) {
    long n = (long) ((size - 2) * (size - 2));
    long floor = n / (long) n_threads;
    for (size_t i = 0; i < n_threads; i++) {
        writeBatchLengths[i] = floor;
    }
    writeBatchLengths[0] += n % (long) n_threads;
}

bool updateIndex(long i, long j, double** mat, double** copy, pthread_mutex_t mtx) {
    double meanValues[] = {
        copy[i - 1][j],
        copy[i][j + 1],
        copy[i + 1][j],
        copy[i][j - 1],
    };
    
    pthread_mutex_lock(&mtx);
    mat[i][j] = doubleMean(meanValues, 4);
    pthread_mutex_unlock(&mtx);

    bool stop = fabs(mat[i][j] - copy[i][j]) < PRECISION;
    return stop;
}

void* manageReadThread(void* voidBatch) {
    ReadBatch* batch = (ReadBatch*) voidBatch;
    long i, j;
    for (size_t k = 0; k < batch->batchLength; k++) {
        i = batch->matrixLocations[k].i;
        j = batch->matrixLocations[k].j;
        batch->copy[i][j] = batch->mat[i][j];
    }
    return NULL;
}

void* manageWriteThread(void* voidBatch) {
    WriteBatch* batch = (WriteBatch*) voidBatch;
    long i, j;
    for (size_t k = 0; k < batch->batchLength; k++) {
        i = batch->matrixLocations[k].i;
        j = batch->matrixLocations[k].j;
        if (!updateIndex(i, j, batch->mat, batch->copy, batch->mtxMat[i][j])) {
            batch->stop = false;
        }
    }
    return NULL;
}

void doubleMatrixDeepCopy(double** mat, double** copy, size_t n_threads, MatrixLocation** batchMatrixLocations, long* batchLengths) {
    pthread_t threads[n_threads];
    ReadBatch* batches[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        batches[i] = (ReadBatch*) malloc(sizeof(ReadBatch));
        batches[i]->batchLength = (size_t) batchLengths[i];
        batches[i]->matrixLocations = batchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->copy = copy;
        assert(pthread_create(&threads[i], NULL, manageReadThread, (void*) batches[i]) == 0);
    }
    for (size_t i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
        free(batches[i]);
    }
}

MatrixLocation** initBatchMatrixLocations(long* batchLengths, size_t n_threads) {
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) malloc(n_threads * sizeof(MatrixLocation*));
    for (size_t i = 0; i < n_threads; i++) {
        batchMatrixLocations[i] = (MatrixLocation*) malloc((size_t) batchLengths[i] * sizeof(MatrixLocation));
    }
    return batchMatrixLocations;
}

void freeBatchMatrixLocations(MatrixLocation** batchMatrixLocations, size_t n_threads) {
    for (size_t i = 0; i < n_threads; i++) {
        free(batchMatrixLocations[i]);
    }
    free(batchMatrixLocations);
}

void calculateReadBatchMatrixLocations(MatrixLocation** readBatchMatrixLocations, long* readBatchLengths, size_t size) {
    size_t batchNumber = 0;
    size_t batchProcessed = 0;
    for (long i = 0; i < (long) size; i++) {
        for (long j = 0; j < (long) size; j++) {
            readBatchMatrixLocations[batchNumber][batchProcessed].i = i;
            readBatchMatrixLocations[batchNumber][batchProcessed].j = j;
            if (++batchProcessed == (size_t) readBatchLengths[batchNumber]) {
                batchNumber++;
                batchProcessed = 0;
            }
        }
    }
}

void calculateWriteBatchMatrixLocations(MatrixLocation** writeBatchMatrixLocations, long* writeBatchLengths, size_t size) {
    size_t batchNumber = 0;
    size_t batchProcessed = 0;
    for (long i = 1; i < (long) size - 1; i++) {
        for (long j = 1; j < (long) size - 1; j++) {
            writeBatchMatrixLocations[batchNumber][batchProcessed].i = i;
            writeBatchMatrixLocations[batchNumber][batchProcessed].j = j;
            if (++batchProcessed == (size_t) writeBatchLengths[batchNumber]) {
                batchNumber++;
                batchProcessed = 0;
            }
        }
    }
}

bool relaxationStep(double** mat, double** copy, size_t size, long* readBatchLengths, MatrixLocation** readBatchMatrixLocations, long* writeBatchLengths, MatrixLocation** writeBatchMatrixLocations, pthread_mutex_t** mtxMat, pthread_t* threads, size_t n_threads) {
    bool stop = true;
    doubleMatrixDeepCopy(mat, copy, n_threads, readBatchMatrixLocations, readBatchLengths);
    WriteBatch* batches[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        batches[i] = (WriteBatch*) malloc(sizeof(WriteBatch));
        batches[i]->batchLength = (size_t) writeBatchLengths[i];
        batches[i]->matrixLocations = writeBatchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->mtxMat = mtxMat;
        batches[i]->copy = copy;
        batches[i]->stop = true;
        assert(pthread_create(&threads[i], NULL, manageWriteThread, (void*) batches[i]) == 0);
    }
    for (size_t i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
        if (!batches[i]->stop) {
            stop = false;
        }
        free(batches[i]);
    }

    return stop;
}

void relaxation(double** mat, size_t size, size_t n_threads, bool logging) {
    long* readBatchLengths = (long*) calloc(n_threads, sizeof(long));
    calculateReadBatchLengths(readBatchLengths, size, n_threads);
    MatrixLocation** readBatchMatrixLocations = (MatrixLocation**) initBatchMatrixLocations(readBatchLengths, n_threads);
    calculateReadBatchMatrixLocations(readBatchMatrixLocations, readBatchLengths, size);

    long* writeBatchLengths = (long*) calloc(n_threads, sizeof(long));
    calculateWriteBatchLengths(writeBatchLengths, size, n_threads);
    MatrixLocation** writeBatchMatrixLocations = (MatrixLocation**) initBatchMatrixLocations(writeBatchLengths, n_threads);
    calculateWriteBatchMatrixLocations(writeBatchMatrixLocations, writeBatchLengths, size);

    double** copy = initDoubleMatrix(size);
    pthread_t threads[n_threads];
    pthread_mutex_t** mtxMat = initMutexMatrix(size);

    bool stop = false;
    if (logging) logSquareDoubleMatrix(mat, size);
    while (!stop) {
        stop = relaxationStep(mat, copy, size, readBatchLengths, readBatchMatrixLocations, writeBatchLengths, writeBatchMatrixLocations, mtxMat, threads, n_threads);
        if (logging) logSquareDoubleMatrix(mat, size);
    }

    freeDoubleMatrix(copy);
    free(writeBatchLengths);
    freeBatchMatrixLocations(writeBatchMatrixLocations, n_threads);
    free(readBatchLengths);
    freeBatchMatrixLocations(readBatchMatrixLocations, n_threads);
    freeMutexMatrix(mtxMat);
}

int main(int argc, char** argv) {
    char* dataFilePath = argv[1];
    size_t n_threads = (size_t) atol(argv[2]);
    size_t size = 0;
    FILE* dataFile = fopen(dataFilePath, "r");

    fscanf(dataFile, "%ld", &size);
    double** mat = initDoubleMatrix(size);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            fscanf(dataFile, "%lf", &mat[i][j]);
        }
    }

    fclose(dataFile);

    struct timespec start, stop, delta;

    clock_gettime(CLOCK_REALTIME, &start);
    relaxation(mat, size, n_threads, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);

    double duration = doubleTime(delta);

    logDuration(size, duration, n_threads);
    freeDoubleMatrix(mat);
    return 0;
}
