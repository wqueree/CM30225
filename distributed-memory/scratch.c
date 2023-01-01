#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "utils.h"

int main() {
    int n = 4;
    int m = 4;
    double** mat = initDoubleMatrix(n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            double n = (double) i * j;
            mat[i][j] = (double) (rand() % 100);
        }
    }

    logDoubleMatrix(mat, n, m);
    double* flattened = flattenRows(mat, 1, 3, 4);

    for (size_t i = 0; i < (3 - 1) * 4; i++) {
        printf("%.2lf ", flattened[i]);
    }
    printf("\n");

    return 0;
}
