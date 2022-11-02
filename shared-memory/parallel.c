#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define SIZE 4
#define PRECISION 0.001
#define THREADS 2

typedef struct ArrayLocation {
    long i;
    long j;
} ArrayLocation;

typedef struct RelaxationBatch {
    long batchLength;
    double* start;
    double* batchStart;
    double* array;
    double* temp;
    bool stop;
} RelaxationBatch;

void logSquareDoubleArray(double array[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
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

long* calculateThreadLengths() {
    long n = (SIZE - 2) * (SIZE - 2);
    long* threadLengths = (long*) malloc(THREADS * sizeof(long));
    memset(threadLengths, 0, THREADS * sizeof(*threadLengths));
    long floor = n / THREADS;
    for (long i = 0; i < THREADS; i++) {
        threadLengths[i] = floor;
    }
    threadLengths[0] += n % THREADS;
    return threadLengths;
}

bool updateIndex(long i, long j, double* array, double* temp) {
    double meanValues[] = {
        temp[((i - 1) * SIZE) + j],
        temp[(i * SIZE) + j + 1],
        temp[((i + 1) * SIZE) + j],
        temp[(i * SIZE) + j - 1],
    };
    array[(i * SIZE) + j] = doubleMean(meanValues, 4);
    bool stop = fabs(array[(i * SIZE) + j] - temp[(i * SIZE) + j]) < PRECISION;
    return stop;
}

ArrayLocation* calculateArrayLocation(double* start, double* current) {
    long bytesFromStart = ((long) current - (long) start);
    long longsFromStart = bytesFromStart / sizeof(long);
    ArrayLocation* arrayLocation = (ArrayLocation*) malloc(sizeof(ArrayLocation));
    arrayLocation->i = longsFromStart / SIZE;
    arrayLocation->j = longsFromStart % SIZE;
    return arrayLocation;
}

void* manageThread(void* voidBatch) {
    RelaxationBatch* batch = (RelaxationBatch*) voidBatch;
    double* current = batch->batchStart;
    for (long i = 0; i < batch->batchLength; i++) {
        current += sizeof(long);
        ArrayLocation* arrayLocation = calculateArrayLocation(batch->start, current);
        if (!updateIndex(arrayLocation->i, arrayLocation->j, batch->array, batch->temp)) {
            batch->stop = false;
        }
    }
    return NULL;
}

bool relaxationStep(double array[SIZE][SIZE]) {
    double temp[SIZE][SIZE];
    memcpy(temp, array, sizeof(double) * SIZE * SIZE);
    double* start = &temp[0][0];
    bool stopIteration = true;
    pthread_t threads[THREADS];
    long batchResults[THREADS];
    long* batchLengths = calculateThreadLengths();
    long processed = 0;
    for (int i = 0; i < THREADS; i++) {
        double* batchStart = (double*) ((long) start + ((SIZE + 1 + processed) * sizeof(long)));
        RelaxationBatch* batch = (RelaxationBatch*) malloc(sizeof(RelaxationBatch));
        batch->batchLength = batchLengths[i];
        batch->start = start;
        batch->batchStart = batchStart;
        batch->array = &array[0][0];
        batch->temp = &temp[0][0];
        batch->stop = true;
        batchResults[i] = pthread_create(&threads[i], NULL, manageThread, (void*) batch);
        processed += batchLengths[i];
    }
    for (int i = 0; i < THREADS; i++) {
        if (!batchResults[i]) {
            stopIteration = false;
        }
    }
    for (int i = 0; i < THREADS; i++) {
        free(&batchLengths[i]);
    }
    return stopIteration;
}

void relaxation(double array[SIZE][SIZE]) {
    bool stopIteration = false;
    while (!stopIteration) {
        logSquareDoubleArray(array);
        stopIteration = relaxationStep(array);
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
