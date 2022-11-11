#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

long* calculateBatchLengths(size_t size, size_t n_threads) {
    long n = (long) ((size - 2) * (size - 2));
    long* batchLengths = (long*) calloc(n_threads, sizeof(long));
    long floor = n / (long) n_threads;
    for (size_t i = 0; i < n_threads; i++) {
        batchLengths[i] = floor;
    }
    batchLengths[0] += n % (long) n_threads;
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

bool relaxationStep(double** mat, size_t size, pthread_mutex_t mat_mtx, size_t n_threads) {
    double** temp = doubleMatrixDeepCopy(mat, size);
    bool stop = true;
    pthread_t threads[n_threads];
    long* batchLengths = calculateBatchLengths(size, n_threads);
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) initBatchMatrixLocations(batchLengths, n_threads);
    calculateBatchMatrixLocations(batchMatrixLocations, batchLengths, size);
    RelaxationBatch* batches[n_threads];

    for (size_t i = 0; i < n_threads; i++) {
        batches[i] = (RelaxationBatch*) malloc(sizeof(RelaxationBatch));
        batches[i]->batchLength = (size_t) batchLengths[i];
        batches[i]->matrixLocations = batchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->mat_mtx = mat_mtx;
        batches[i]->temp = temp;
        batches[i]->stop = true;
        assert(pthread_create(&threads[i], NULL, manageThread, (void*) batches[i]) == 0);
    }
    for (size_t i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
        if (!batches[i]->stop) {
            stop = false;
        }
        free(batches[i]);
    }
    freeDoubleMatrix(temp);
    free(batchLengths);
    freeBatchMatrixLocations(batchMatrixLocations, n_threads);
    return stop;
}

void relaxation(double** mat, size_t size, pthread_mutex_t mat_mtx, size_t n_threads, bool logging) {
    bool stop = false;
    if (logging) logSquareDoubleMatrix(mat, size);
    while (!stop) {
        stop = relaxationStep(mat, size, mat_mtx, n_threads);
        if (logging) logSquareDoubleMatrix(mat, size);
    }
}

int main(int argc, char** argv) {
    char* dataFilePath = argv[1];
    size_t n_threads = (size_t) atol(argv[2]);

    FILE* dataFile = fopen(dataFilePath, "r");

    size_t size = 0;

    fscanf(dataFile, "%ld", &size);

    double** mat = initDoubleMatrix(size);
    // pthread_mutex_t** mtxMat = initMutexMatrix(size);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            fscanf(dataFile, "%lf", &mat[i][j]);
        }
    }

    fclose(dataFile);

    pthread_mutex_t mat_mtx;
    pthread_mutex_init(&mat_mtx, NULL);

    struct timespec start, stop, delta;

    clock_gettime(CLOCK_REALTIME, &start);
    relaxation(mat, size, mat_mtx, n_threads, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);

    double duration = doubleTime(delta);

    logDuration(size, duration, n_threads);
    freeDoubleMatrix(mat);
    // freeMutexMatrix(mtxMat);
    return 0;
}
