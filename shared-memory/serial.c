#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define PRECISION 0.01
#define THREADS 1

void logSquareDoubleMatrix(double** mat, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** initDoubleMatrix(size_t size) {
    double** mat = (double**) malloc(size * sizeof(double*));
    double* matBuf = malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        mat[i] = (size * i) + matBuf;
    }
    return mat;
}

void freeDoubleMatrix(double** mat) {
    free(mat[0]);
    free(mat);
}

double** doubleMatrixDeepCopy(double** mat, size_t size) {
    double** copy = (double**) initDoubleMatrix(size);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            copy[i][j] = mat[i][j];
        }
    }
    return copy;
}

double doubleMean(double mat[], int n) {
    double matSum = 0.0;
    for (int i = 0; i < n; i++) {
        matSum += mat[i];
    }
    return matSum / n;
}

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

void relaxation(double** mat, size_t size) {
    bool stop = false;
    logSquareDoubleMatrix(mat, size);
    while (!stop) {
        stop = relaxationStep(mat, size);
        logSquareDoubleMatrix(mat, size);
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
    
    relaxation(mat, size);
    freeDoubleMatrix(mat);
    return 0;
}