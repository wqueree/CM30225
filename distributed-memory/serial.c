#include <stdbool.h>
#include <math.h>
#include "utils.h"

bool relaxationStep(float** mat, float** cpy, size_t size) {
    // Completes one step of the relaxation method.
    // squareFloatMatrixDeepCopy(mat, cpy, size);
    bool stop = true;
    for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            cpy[i][j] = calculateNeighbourMean(mat, i, j);
            if (fabs(mat[i][j] - cpy[i][j]) > PRECISION) {
                stop = false; // At least one element in the matrix is outside PRECISION
            }
        }
    }
    return stop;
}

void relaxation(float** mat, size_t size, bool logging) {
    bool stop = false;
    float** cpy = initSquareFloatMatrix(size);
    squareFloatMatrixDeepCopy(mat, cpy, size);
    if (logging) logSquareFloatMatrix(mat, size);
    while (!stop) { // While values are outside of PRECISION
        stop = relaxationStep(mat, cpy, size);
        matrixSwap(&mat, &cpy);
        if (logging) logSquareFloatMatrix(mat, size);
    }
    logSquareFloatMatrix(mat, size);
    freeFloatMatrix(mat);
    freeFloatMatrix(cpy);
}

int main(int argc, char** argv) {
    // Should be invoked from command line as follows:
    // ./serial path/to/test/file.txt
    char* dataFilePath = argv[1];
    FILE* dataFile = fopen(dataFilePath, "r");

    // File IO
    size_t size = 0;

    fscanf(dataFile, "%ld", &size);

    float** mat = initSquareFloatMatrix(size);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            fscanf(dataFile, "%f", &mat[i][j]);
        }
    }

    fclose(dataFile);

    // Timing
    struct timespec start, stop, delta;

    clock_gettime(CLOCK_REALTIME, &start);
    relaxation(mat, size, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);
    float duration = floatTime(delta);

    logDuration(size, duration, 0);
    return 0;
}