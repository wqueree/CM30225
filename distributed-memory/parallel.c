#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

void calculateProcessorChunks(long* processorChunks, size_t size, size_t n_chunks) {
    long floor = size / (long) n_chunks;
    for (size_t i = 0; i < n_chunks - 1; i++) {
        processorChunks[i] = floor;
    }
    processorChunks[n_chunks - 1] += size % (long) n_chunks;
}

void relaxation(double** mat, size_t size, size_t n_processors, int mpi_rank, bool logging) {
    bool stop = false;
    while (!stop) {
        if (mpi_rank == 0) {
            size_t n_chunks = n_processors - 1;
            long processorChunks[n_chunks];
            calculateProcessorChunks(processorChunks, size, n_chunks);
            // Send chunks to worker processors
        }
        else {
            // Receive chunk
            // Process
            // Send back to 0
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
    double** mat = initDoubleMatrix(size);

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

    int mpi_rank, mpi_size, nproc, namelen;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 

    printf("%d\n", mpi_rank);
    if (mpi_rank == 0) {
        printf("main reports %d procs\n", nproc);
    }
    // namelen = MPI_MAX_PROCESSOR_NAME;
    // MPI_Get_processor_name(name, &namelen);
    // printf("hello world %d from %s\n", mpi_rank, name);

    MPI_Finalize();

    relaxation(mat, size, n_processors, mpi_rank, LOGGING);

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
