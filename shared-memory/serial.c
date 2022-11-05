#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define SIZE 4
#define PRECISION 0.001
#define THREADS 1

void logSquareDoubleMatrix(double mat[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double doubleMean(double array[], int n) {
    double arraySum = 0.0;
    for (int i = 0; i < n; i++) {
        arraySum += array[i];
    }
    return arraySum / n;
}

bool relaxationStep(double array[SIZE][SIZE]) {
    double temp[SIZE][SIZE];
    bool validDelta = true;
    memcpy(temp, array, sizeof(double) * SIZE * SIZE);
    for (int i = 1; i < SIZE - 1; i++) {
        for (int j = 1; j < SIZE - 1; j++) {
            double meanValues[] = {
                temp[i - 1][j],
                temp[i][j + 1],
                temp[i + 1][j],
                temp[i][j - 1]
            };
            array[i][j] = doubleMean(meanValues, 4);
            if (fabs(array[i][j] - temp[i][j]) > PRECISION) {
                validDelta = false;
            }
        }
    }
    return validDelta;
}

void relaxation(double array[SIZE][SIZE]) {
    bool validDelta = false;
    logSquareDoubleMatrix(array);
    while (!validDelta) {
        validDelta = relaxationStep(array);
        logSquareDoubleMatrix(array);
    }
}

int main() {
    double example[SIZE][SIZE] = {
        {1.0, 1.0, 1.0, 1.0}, 
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };
    relaxation(example);
    return 0;
}