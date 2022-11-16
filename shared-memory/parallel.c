#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

void calculateReadBatchLengths(long* readBatchLengths, size_t size, size_t n_threads) {
    // Splits the full matrix into n_threads batches and stores them in readBatchLengths
    long n = (long) (size * size);
    long floor = n / (long) n_threads;
    for (size_t i = 0; i < n_threads; i++) {
        readBatchLengths[i] = floor;
    }
    readBatchLengths[0] += n % (long) n_threads;
}

void calculateWriteBatchLengths(long* writeBatchLengths, size_t size, size_t n_threads) {
    // Splits the inner matrix into n_threads batches and stores them in writeBatchLengths
    long n = (long) ((size - 2) * (size - 2));
    long floor = n / (long) n_threads;
    for (size_t i = 0; i < n_threads; i++) {
        writeBatchLengths[i] = floor;
    }
    writeBatchLengths[0] += n % (long) n_threads;
}

bool updateIndex(long i, long j, double** mat, double** copy) {
    // Calculates the mean for an element index from its neighbours and updates in mat
    double meanValues[] = {
        copy[i - 1][j],
        copy[i][j + 1],
        copy[i + 1][j],
        copy[i][j - 1],
    };
    
    mat[i][j] = doubleMean(meanValues, 4);

    // true if difference beween old and new value is within PRECISION
    bool stop = fabs(mat[i][j] - copy[i][j]) < PRECISION;
    return stop;
}

void* manageReadThread(void* voidBatch) {
    // Manages the copies from mat into copy using pthreads for some batch voidBatch
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
    // Manages the updates of mat using copy and pthreads for some voidBatch
    WriteBatch* batch = (WriteBatch*) voidBatch;
    long i, j;
    for (size_t k = 0; k < batch->batchLength; k++) {
        i = batch->matrixLocations[k].i;
        j = batch->matrixLocations[k].j;
        if (!updateIndex(i, j, batch->mat, batch->copy)) {
            batch->stop = false; // At least one element in batch is outside of PRECISION
        }
    }
    return NULL;
}

void doubleMatrixDeepCopy(double** mat, double** copy, size_t n_threads, MatrixLocation** batchMatrixLocations, long* batchLengths) {
    // Manages the work distribution for the multithreaded matrix copy operation
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
    // Allocates memory for the set of locations in the matrix for each batch (and hence each thread)
    MatrixLocation** batchMatrixLocations = (MatrixLocation**) malloc(n_threads * sizeof(MatrixLocation*));
    for (size_t i = 0; i < n_threads; i++) {
        batchMatrixLocations[i] = (MatrixLocation*) malloc((size_t) batchLengths[i] * sizeof(MatrixLocation));
    }
    return batchMatrixLocations;
}

void freeBatchMatrixLocations(MatrixLocation** batchMatrixLocations, size_t n_threads) {
    // Frees memory allocated by initBatchMatrixLocations
    for (size_t i = 0; i < n_threads; i++) {
        free(batchMatrixLocations[i]);
    }
    free(batchMatrixLocations);
}

void calculateReadBatchMatrixLocations(MatrixLocation** readBatchMatrixLocations, long* readBatchLengths, size_t size) {
    // Calculates matrix locations for full matrix and writes to readBatchMatrixLocations
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
    // Calculates matrix locations for inner matrix and writes to writeBatchMatrixLocations
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

bool relaxationStep(double** mat, double** copy, size_t size, long* readBatchLengths, MatrixLocation** readBatchMatrixLocations, long* writeBatchLengths, MatrixLocation** writeBatchMatrixLocations, pthread_t* threads, size_t n_threads) {
    // Completes one iteration of the relaxation method in parallel
    bool stop = true; // Indicates if iteration should be stopped (if all values are within PRECISION of previous)
    doubleMatrixDeepCopy(mat, copy, n_threads, readBatchMatrixLocations, readBatchLengths);
    WriteBatch* batches[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        batches[i] = (WriteBatch*) malloc(sizeof(WriteBatch));
        batches[i]->batchLength = (size_t) writeBatchLengths[i];
        batches[i]->matrixLocations = writeBatchMatrixLocations[i];
        batches[i]->mat = mat;
        batches[i]->copy = copy;
        batches[i]->stop = true;
        assert(pthread_create(&threads[i], NULL, manageWriteThread, (void*) batches[i]) == 0);
    }
    for (size_t i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
        if (!batches[i]->stop) {
            stop = false; // At least one matrix in batch is outside of PRECISION
        }
        free(batches[i]);
    }

    return stop;
}

void relaxation(double** mat, size_t size, size_t n_threads, bool logging) {
    // Executes the relaxation method on mat. Sets up data structures to do so.
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

    bool stop = false;
    if (logging) logSquareDoubleMatrix(mat, size);
    while (!stop) { // Until all values of the matrix are within PRECISION
        stop = relaxationStep(mat, copy, size, readBatchLengths, readBatchMatrixLocations, writeBatchLengths, writeBatchMatrixLocations, threads, n_threads);
        if (logging) logSquareDoubleMatrix(mat, size);
    }

    freeDoubleMatrix(copy);
    free(writeBatchLengths);
    freeBatchMatrixLocations(writeBatchMatrixLocations, n_threads);
    free(readBatchLengths);
    freeBatchMatrixLocations(readBatchMatrixLocations, n_threads);
}

int main(int argc, char** argv) {
    // Main function. Should be invoked from command line as follows:
    // ./parallel path/to/matrix/file.txt number-of-threads
    // An example matrix file is attached in 8.txt
    char* dataFilePath = argv[1];
    size_t n_threads = (size_t) atol(argv[2]);

    // File IO
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

    // Timing
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
