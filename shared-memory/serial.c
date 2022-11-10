#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "utils.h"

bool relaxationStep(double** mat, size_t size) {
    double** temp = doubleMatrixDeepCopy(mat, size);
    bool stop = true;
    for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            double meanValues[] = {
                temp[i - 1][j],
                temp[i][j + 1],
                temp[i + 1][j],
                temp[i][j - 1]
            };
            mat[i][j] = doubleMean(meanValues, 4);
            if (fabs(mat[i][j] - temp[i][j]) > PRECISION) {
                stop = false;
            }
        }
    }
    freeDoubleMatrix(temp);
    return stop;
}

void relaxation(double** mat, size_t size, bool logging) {
    bool stop = false;
    if (logging) logSquareDoubleMatrix(mat, size);
    while (!stop) {
        stop = relaxationStep(mat, size);
        if (logging) logSquareDoubleMatrix(mat, size);
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

    struct timespec start, stop, delta;

    clock_gettime(CLOCK_REALTIME, &start);
    relaxation(mat, size, LOGGING);
    clock_gettime(CLOCK_REALTIME, &stop);

    timespecDifference(start, stop, &delta);
    double duration = doubleTime(delta);

    logDuration(size, duration);
    freeDoubleMatrix(mat);
    return 0;
}