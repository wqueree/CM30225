#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define N 4
#define T 0.001

void logSquareDoubleArray(double array[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", array[i][j]);
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

bool relaxationStep(double array[N][N]) {
    double temp[N][N];
    bool validDelta = true;
    memcpy(temp, array, sizeof(double) * N * N);
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            double meanValues[] = {
                temp[i - 1][j],
                temp[i][j + 1],
                temp[i + 1][j],
                temp[i][j - 1]
            };
            array[i][j] = doubleMean(meanValues, 4);
            if (fabs(array[i][j] - temp[i][j]) > T) {
                validDelta = false;
            }
        }
    }
    return validDelta;
}

void relaxation(double array[N][N]) {
    bool validDelta = false;
    while (!validDelta) {
        logSquareDoubleArray(array);
        validDelta = relaxationStep(array);
    }
}

int main() {
    double example[4][4] = {
        {1.0, 1.0, 1.0, 1.0}, 
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };
    relaxation(example);
    return 0;
}