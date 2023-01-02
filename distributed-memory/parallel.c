#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

void calculateProcessorChunkSizes(long* processorChunkRows, size_t size, size_t n_chunks) {
    long floor = size / (long) n_chunks;
    for (size_t i = 0; i < n_chunks; i++) {
        processorChunkRows[i] = floor;
    }
    processorChunkRows[n_chunks - 1] += size % (long) n_chunks;
}

void calculateProcessorChunkRows(long* processorChunkRows, long* processorChunkSizes, size_t n_chunks) {
    processorChunkRows[0] = 0;
    for (size_t i = 1; i < n_chunks; i++) {
        processorChunkRows[i] = processorChunkRows[i - 1] + processorChunkSizes[i - 1];
    }
}

void generateProcessorChunks(FlatMatrixChunk* processorChunks, long* processorChunkRows, double** mat, size_t n_chunks, size_t n, size_t m) {
    for (size_t i = 0; i < n_chunks - 1; i++) {
        processorChunks[i] = *flattenRows(mat, processorChunkRows[i], processorChunkRows[i + 1], m);
    }
    processorChunks[n_chunks - 1] = *flattenRows(mat, processorChunkRows[n_chunks - 1], n, m);
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

void relaxation(double** mat, size_t size, size_t n_processors, int mpi_rank, bool logging) {
    bool stop = false;
    while (!stop) {
        if (mpi_rank == 0) {
            size_t n_chunks = n_processors - 1;
            long processorChunkSizes[n_chunks];
            long processorChunkRows[n_chunks];
            FlatMatrixChunk processorChunks[n_chunks];
            calculateProcessorChunkSizes(processorChunkSizes, size, n_chunks);
            calculateProcessorChunkRows(processorChunkRows, processorChunkSizes, n_chunks);
            generateProcessorChunks(processorChunks, processorChunkRows, mat, n_chunks, size, size);

            // Distribute to worker cores
            for (size_t i = 0; i < n_chunks; i++) {
                FlatMatrixChunk chunk = processorChunks[i];
                int worker = i + 1;
                long sizeBuf[] = {chunk.n, chunk.m};
                MPI_Send(sizeBuf, 2, MPI_LONG, worker, 0, MPI_COMM_WORLD);
                MPI_Send(chunk.flat, chunk.n * chunk.m, MPI_DOUBLE, worker, 1, MPI_COMM_WORLD);
            }
            // Receive completed computations from worker cores
            for (size_t i = 0; i < n_chunks; i++) {
                FlatMatrixChunk chunk = processorChunks[i];
                int worker = i + 1;
                MPI_Recv(chunk.flat, chunk.n * chunk.m, MPI_DOUBLE, worker, 2, MPI_COMM_WORLD, 0);
                // printf("\n");
                // for (size_t j = 0; j < chunk.n * chunk.m; j++) {
                //     printf("%.2lf ", chunk.flat[j]);
                // }
                // printf("\n");
            }
            double** cpy = initSquareDoubleMatrix(size);
            for (size_t i = 0; i < n_chunks; i++) {
                FlatMatrixChunk chunk = processorChunks[i];
                for (size_t j = chunk.start_row; j < chunk.start_row + chunk.n; j++) {
                    for (size_t k = 0; k < chunk.m; k++) {
                        cpy[j][k] = chunk.flat[((j - chunk.start_row) * chunk.m) + k];
                    }
                }
            }

            logSquareDoubleMatrix(mat, size);
            logSquareDoubleMatrix(cpy, size);
            stop = true; // Only one iteration.sbat
        }
        else {
            long sizeBuf[2];
            MPI_Recv(sizeBuf, 2, MPI_LONG, 0, 0, MPI_COMM_WORLD, 0);
            size_t n = sizeBuf[0];
            size_t m = sizeBuf[1];
            double flat[n * m];
            MPI_Recv(flat, n * m, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, 0);
            double** chunk = reshapeRows(flat, n, m);
            double** result = initDoubleMatrix(n, m);
            doubleMatrixDeepCopy(chunk, result, n, m);
            for (size_t i = 1; i < n - 1; i++) {
                for (size_t j = 1; j < m - 1; j++) {
                    double meanValues[] = {
                        chunk[i - 1][j],
                        chunk[i][j + 1],
                        chunk[i + 1][j],
                        chunk[i][j - 1]
                    };
                    result[i][j] = doubleMean(meanValues, 4);
                }
            }

            // logDoubleMatrix(chunk, n, m);
            // logDoubleMatrix(result, n ,m);

            FlatMatrixChunk* resultFlatMatrixChunk = flattenRows(result, 0, n, m);
            double* flatResult = resultFlatMatrixChunk->flat;

            // printf("\n");
            // for (size_t j = 0; j < n * m; j++) {
            //     printf("%.2lf ", flatResult[j]);
            // }
            // printf("\n");
            MPI_Send(flatResult, n * m, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            freeDoubleMatrix(chunk);
            freeDoubleMatrix(result);
            free(resultFlatMatrixChunk);
            free(flatResult);
            stop = true; // Only one iteration
        }
    }
}

int main(int argc, char** argv) {
    // Main function. Should be invoked from command line as follows:
    // ./parallel path/to/matrix/file.txt number-of-threads
    // An example matrix file is attached in 8.txt
    char* dataFilePath = argv[1];
    size_t n_processors = (size_t) atol(argv[2]);

    // File IO
    size_t size = 0;
    FILE* dataFile = fopen(dataFilePath, "r");

    fscanf(dataFile, "%ld", &size);
    double** mat = initSquareDoubleMatrix(size);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            fscanf(dataFile, "%lf", &mat[i][j]);
        }
    }
    fclose(dataFile);

    // MPI Setup
    char name[MPI_MAX_PROCESSOR_NAME];

    int mpi_init_rc = MPI_Init(&argc, &argv);
    if (mpi_init_rc != MPI_SUCCESS) {
        printf ("Error starting MPI program.\n");
        MPI_Abort(MPI_COMM_WORLD, mpi_init_rc);
    }

    int mpi_rank, mpi_size, namelen;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 

    if (mpi_rank == 0) {
        printf("main reports %d procs\n", mpi_size);
    }
    relaxation(mat, size, mpi_size, mpi_rank, LOGGING);

    // namelen = MPI_MAX_PROCESSOR_NAME;
    // MPI_Get_processor_name(name, &namelen);
    // printf("hello world %d from %s\n", mpi_rank, name);

    MPI_Finalize();


    // Timing

    // while (!stop) {
    //     if (mpi_rank == 0) {
    // struct timespec start, stop, delta;
    //         clock_gettime(CLOCK_REALTIME, &start);
    //         relaxation(mat, size, n_processors, LOGGING);

    //         clock_gettime(CLOCK_REALTIME, &stop);

    //         timespecDifference(start, stop, &delta);

    //         double duration = doubleTime(delta);

    //         logDuration(size, duration, n_processors);
    //         freeDoubleMatrix(mat);
    //     }
    // }
    return 0;
}
