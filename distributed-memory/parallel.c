#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

void flattenRows(FlatMatrixChunk* flatMatrixChunk, double** mat, size_t start_row, size_t end_row, size_t row_size) {
    // end row terminates iteration and is not included.
    size_t num_rows = end_row - start_row;
    double* flat = (double*) malloc(num_rows * row_size * sizeof(double));
    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < row_size; j++) {
            flat[((i - start_row) * row_size) + j] = mat[i][j];
        }
    }
    flatMatrixChunk->n = num_rows;
    flatMatrixChunk->m = row_size;
    flatMatrixChunk->start_row = start_row;
    flatMatrixChunk->flat = flat;
}

void freeFlatMatrixChunk(FlatMatrixChunk* flatMatrixChunk) {
    free(flatMatrixChunk->flat);
}

void calculateProcessorChunkSizes(long* processorChunkRows, size_t size, size_t n_chunks) {
    long floor = (long) size / (long) n_chunks;
    for (size_t i = 0; i < n_chunks; i++) {
        processorChunkRows[i] = floor;
    }
    processorChunkRows[n_chunks - 1] += (long) size % (long) n_chunks;
}

void calculateProcessorChunkRows(long* processorChunkRows, long* processorChunkSizes, size_t n_chunks) {
    processorChunkRows[0] = 0;
    for (size_t i = 1; i < n_chunks; i++) {
        processorChunkRows[i] = processorChunkRows[i - 1] + processorChunkSizes[i - 1];
    }
}

void generateProcessorChunks(FlatMatrixChunk* processorChunks, long* processorChunkRows, double** mat, size_t n_chunks, size_t n, size_t m) {
    for (size_t i = 0; i < n_chunks - 1; i++) {
        flattenRows(&processorChunks[i], mat, (size_t) processorChunkRows[i], (size_t) processorChunkRows[i + 1], m);
    }
    flattenRows(&processorChunks[n_chunks - 1], mat, (size_t) processorChunkRows[n_chunks - 1], n, m);
}

void distributeChunks(FlatMatrixChunk* processorChunks, size_t n_chunks) {
    for (size_t i = 0; i < n_chunks; i++) {
        FlatMatrixChunk chunk = processorChunks[i];
        int worker = (int) i + 1;
        long sizeBuf[] = {(int) chunk.n, (int) chunk.m};
        MPI_Send(sizeBuf, 2, MPI_LONG, worker, 0, MPI_COMM_WORLD);
        MPI_Send(chunk.flat, (int) chunk.n * (int) chunk.m, MPI_DOUBLE, worker, 1, MPI_COMM_WORLD);
    }
}

void collateChunks(FlatMatrixChunk* processorChunks, size_t n_chunks) {
    for (size_t i = 0; i < n_chunks; i++) {
        FlatMatrixChunk chunk = processorChunks[i];
        int worker = (int) i + 1;
        MPI_Recv(chunk.flat, (int) chunk.n * (int) chunk.m, MPI_DOUBLE, worker, 2, MPI_COMM_WORLD, 0);
    }
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

void updateEdgeRows(double** mat, double** cpy, size_t n_chunks, size_t row_size, int mpi_rank, long* processorChunkRows) {
    for (size_t i = 1; i < n_chunks; i++) {
        long leadingEdgeRow = processorChunkRows[i];
        long trailingEdgeRow = leadingEdgeRow - 1;
        for (size_t j = 0; j < row_size; j++) {
            cpy[leadingEdgeRow][j] = calculateNeighbourMean(mat, (size_t) leadingEdgeRow, j);
            cpy[trailingEdgeRow][j] = calculateNeighbourMean(mat, (size_t) trailingEdgeRow, j);
        }
    }
}

void rebuildMatrix(double** cpy, FlatMatrixChunk* processorChunks, size_t n_chunks) {
    for (size_t i = 0; i < n_chunks; i++) {
        FlatMatrixChunk chunk = processorChunks[i];
        for (size_t j = chunk.start_row + 1; j < chunk.start_row + chunk.n - 1; j++) {
            for (size_t k = 0; k < chunk.m; k++) {
                cpy[j][k] = chunk.flat[((j - chunk.start_row) * chunk.m) + k];
            }
        }
    }
}

bool precisionStopCheck(double** mat, double** cpy, size_t size) {
    for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            if (fabs(mat[i][j] - cpy[i][j]) > PRECISION) {
                return false; // At least one element in the matrix is outside PRECISION
            }
        }
    }
    return true;
}

void matrixSwap(double*** mat, double*** cpy) {
    double** tmp = *mat;
    *mat = *cpy;
    *cpy = tmp;
}

void relaxationMaster(double** mat, size_t size, int mpi_rank, size_t n_processors, bool logging) {
    bool stop = false;
    size_t n_chunks = n_processors - 1;
    long processorChunkSizes[n_chunks];
    long processorChunkRows[n_chunks];
    double** cpy = initSquareDoubleMatrix(size);
    for (size_t i = 0; i < size; i++) {
        cpy[0][i] = mat[0][i];
        cpy[size - 1][i] = mat[size - 1][i];
    }
    FlatMatrixChunk processorChunks[n_chunks];
    calculateProcessorChunkSizes(processorChunkSizes, size, n_chunks);
    calculateProcessorChunkRows(processorChunkRows, processorChunkSizes, n_chunks);
    while (!stop) {
        logSquareDoubleMatrix(mat, size);
        generateProcessorChunks(processorChunks, processorChunkRows, mat, n_chunks, size, size);
        distributeChunks(processorChunks, n_chunks); // Distribute to worker cores
        updateEdgeRows(mat, cpy, n_chunks, size, mpi_rank, processorChunkRows); // Update edge rows in chunks that don't have 4 neighbours
        collateChunks(processorChunks, n_chunks); // Receive completed computations from worker cores
        rebuildMatrix(cpy, processorChunks, n_chunks);
        for (size_t i = 0; i < n_chunks; i++) {
            freeFlatMatrixChunk(&processorChunks[i]);
        }
        stop = precisionStopCheck(mat, cpy, size);
        matrixSwap(&mat, &cpy);
        if (stop) {
            for (size_t i = 0; i < n_chunks; i++) {
                long sizeBuf[] = {0, 0};
                MPI_Send(sizeBuf, 2, MPI_LONG, (int) i + 1, 0, MPI_COMM_WORLD);
            }
            logSquareDoubleMatrix(mat, size);
        }
    }
    freeDoubleMatrix(mat);
    freeDoubleMatrix(cpy);
}

void relaxationSlave(int mpi_rank, bool logging) {
    bool stop = false;
    long sizeBuf[2];
    while (!stop) {
        MPI_Recv(sizeBuf, 2, MPI_LONG, 0, 0, MPI_COMM_WORLD, 0);
        size_t n = (size_t) sizeBuf[0];
        size_t m = (size_t) sizeBuf[1];
        if (n > 0 && m > 0) {
            double flat[n * m];
            MPI_Recv(flat, (int) n * (int) m, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, 0);
            double** chunk = reshapeRows(flat, n, m);
            double** result = initDoubleMatrix(n, m);
            doubleMatrixDeepCopy(chunk, result, n, m);
            for (size_t i = 1; i < n - 1; i++) {
                for (size_t j = 1; j < m - 1; j++) {
                    result[i][j] = calculateNeighbourMean(chunk, i, j);
                }
            }
            FlatMatrixChunk resultFlatMatrixChunk;
            flattenRows(&resultFlatMatrixChunk, result, 0, n, m);
            double* flatResult = resultFlatMatrixChunk.flat;
            MPI_Send(flatResult, (int) n * (int) m, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            freeDoubleMatrix(chunk);
            freeDoubleMatrix(result);
            freeFlatMatrixChunk(&resultFlatMatrixChunk);
        }
        else {
            stop = true;
        } 
    }
}

void relaxation(char* dataFilePath, size_t n_processors, int mpi_rank, bool logging) {
    if (mpi_rank == 0) {
        size_t size;
        double** mat = inputDoubleMatrix(dataFilePath, &size);
        struct timespec start, stop, delta;
        clock_gettime(CLOCK_REALTIME, &start);
        relaxationMaster(mat, size, mpi_rank, n_processors, logging);
        clock_gettime(CLOCK_REALTIME, &stop);
        timespecDifference(start, stop, &delta);
        double duration = doubleTime(delta);
        logDuration(size, duration, n_processors);
    }
    else {
        relaxationSlave(mpi_rank, logging);
    }
}

int main(int argc, char** argv) {
    // Main function. Should be invoked from command line as follows:
    // ./parallel path/to/matrix/file.txt
    // An example matrix file is attached in 8.txt
    char* dataFilePath = argv[1];

    // MPI Setup
    int mpi_init_rc = MPI_Init(&argc, &argv);
    if (mpi_init_rc != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, mpi_init_rc);
    }

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    relaxation(dataFilePath, (size_t) mpi_size, mpi_rank, LOGGING);
    MPI_Finalize();
    return 0;
}
