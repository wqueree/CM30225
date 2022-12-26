#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include "utils.h"

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

    MPI_init()

    fclose(dataFile);

    // Timing
    struct timespec start, stop, delta;

    clock_gettime(CLOCK_REALTIME, &start);
    // relaxation(mat, size, n_threads, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);

    double duration = doubleTime(delta);

    logDuration(size, duration, n_threads);
    freeDoubleMatrix(mat);
    return 0;
}
