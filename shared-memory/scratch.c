#include <stdio.h>

int main() {
    double temp[4][4] = {
        {1.0, 1.0, 1.0, 1.0}, 
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0},
    };

    double* tempPtr;
    tempPtr = &(temp[0][0]);

    int SIZE = 4;
    int i = 1;
    int j = 1;
    double meanValues[] = {
        tempPtr[((i - 1) * SIZE) + j],
        tempPtr[(i * SIZE) + j + 1],
        tempPtr[((i + 1) * SIZE) + j],
        tempPtr[(i * SIZE) + j - 1],
    };

    for (int k = 0; k < SIZE; k++) {
        printf("%lf ", meanValues[k]);
    }
    return 0;
}