#include <stdbool.h>
#include <math.h>
#include "utils.h"

void squareDoubleMatrixDeepCopy(double** mat, double** copy, size_t size) {
    // Creates copy of double matrix serially.
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            copy[i][j] = mat[i][j];
        }
    }
}

bool relaxationStep(double** mat, double** copy, size_t size) {
    // Completes one step of the relaxation method.
    squareDoubleMatrixDeepCopy(mat, copy, size);
    bool stop = true;
    for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            double meanValues[] = {
                copy[i - 1][j],
                copy[i][j + 1],
                copy[i + 1][j],
                copy[i][j - 1]
            };
            mat[i][j] = doubleMean(meanValues, 4);
            if (fabs(mat[i][j] - copy[i][j]) > PRECISION) {
                stop = false; // At least one element in the matrix is outside PRECISION
            }
        }
    }
    return stop;
}

void relaxation(double** mat, size_t size, bool logging) {
    bool stop = false;
    double** copy = initDoubleMatrix(size);
    if (logging) logSquareDoubleMatrix(mat, size);
    while (!stop) { // While values are outside of PRECISION
        stop = relaxationStep(mat, copy, size);
        if (logging) logSquareDoubleMatrix(mat, size);
    }
    freeDoubleMatrix(copy);
}

int main(int argc, char** argv) {
    // Should be invoked from command line as follows:
    // ./serial path/to/test/file.txt
    char* dataFilePath = argv[1];
    FILE* dataFile = fopen(dataFilePath, "r");

    // File IO
    size_t size = 0;

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
    relaxation(mat, size, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);
    double duration = doubleTime(delta);

    logDuration(size, duration, 0);
    freeDoubleMatrix(mat);
    return 0;
}