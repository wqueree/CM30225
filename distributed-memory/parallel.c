#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

void flattenRows(float* flat, float** mat, size_t startRow, size_t endRow, size_t rowSize) {
    // Flatten mat into flat.
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t j = 0; j < rowSize; j++) {
            flat[((i - startRow) * rowSize) + j] = mat[i][j];
        }
    }
}

float** reshapeRows(float* flat, size_t n, size_t m) {
    float** mat = initFloatMatrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            mat[i][j] = flat[(i * m) + j];
        }
    }
    return mat;
}

void arraySwap(float** arr, float** cpy) {
    float* tmp = *arr;
    *arr = *cpy;
    *cpy = tmp;
}

float calculateFlatNeighbourMean(float* matFlat, size_t centre, int denominator, size_t rowSize) {
    float neighbours[] = {
        matFlat[centre + rowSize],
        matFlat[centre + 1],
        matFlat[centre - rowSize],
        matFlat[centre - 1]
    };
    return floatMean(neighbours, denominator);
}

void calculateProcessorChunkSizes(int* processorChunkSizes, size_t size, size_t nChunks) {
    // Chunk workload
    for (size_t i = 0; i < nChunks; i++) {
        processorChunkSizes[i] = 0; // Initialise all sizes to 0.
    }
    size_t chunk = 0;
    for (size_t i = 0; i < size; i++) {
        processorChunkSizes[chunk]++; // Add 1 to chunk size at index chunk
        chunk = (chunk + 1) % nChunks; // Move on to next chunk
    }
}

void calculateProcessorChunkRows(int* processorChunkRows, int* processorChunkSizes, size_t nChunks) {
    // Calculates starting row for each chunk.
    processorChunkRows[0] = 0;
    for (size_t i = 1; i < nChunks; i++) {
        processorChunkRows[i] = processorChunkRows[i - 1] + processorChunkSizes[i - 1];
    }
}

void distributeChunkSizes(int* processorChunkRows, size_t nChunks, size_t size) {
    // Sends each chunk size to its worker processor.
    int m = (int) size;
    for (size_t i = 0; i < nChunks; i++) {
        int startRow = (int) processorChunkRows[i];
        int endRow = i == nChunks - 1 ? (int) size : (int) processorChunkRows[i + 1]; // Set endRow to size if its the last chunk, otherwise the next chunk start row minus 1.
        int n = (int) (endRow - startRow);
        int worker = (int) i + 1;
        int sizeBuf[2] = {n, m};
        MPI_Send(sizeBuf, 2, MPI_INT, worker, 0, MPI_COMM_WORLD);
    }
}

void distributeFlatChunks(int* processorChunkRows, size_t nChunks, float* matFlat, size_t size) {
    // Sends each chunk to its worker processor.
    int m = (int) size;
    for (size_t i = 0; i < nChunks; i++) {
        int startRow = (int) processorChunkRows[i];
        int endRow = i == nChunks - 1 ? (int) size : (int) processorChunkRows[i + 1]; // Set endRow to size if its the last chunk, otherwise the next chunk start row minus 1.
        int n = (int) (endRow - startRow);
        int worker = (int) i + 1;
        MPI_Send(&matFlat[startRow * m], n * m, MPI_FLOAT, worker, 1, MPI_COMM_WORLD);
    }
}

void collateFlatChunks(float* cpyFlat, size_t size, int* processorChunkRows, size_t nChunks) {
    int m = (int) size;
    for (size_t i = 0; i < nChunks; i++) {
        int startRow = (int) processorChunkRows[i];
        int endRow = i == nChunks - 1 ? (int) size : (int) processorChunkRows[i + 1]; // Set endRow to size if its the last chunk, otherwise the next chunk start row minus 1.
        int n = endRow - (int) startRow;
        int worker = (int) i + 1;
        float* start = &cpyFlat[m * (startRow + 1)]; // Receive from row 1 of the chunk
        int nValues = m * ((int) n - 2); // End at the penultimate row of the chunk
        MPI_Recv(start, nValues, MPI_FLOAT, worker, 2, MPI_COMM_WORLD, 0);
    }
}

void updateFlatEdgeRows(float* matFlat, float* cpyFlat, size_t nChunks, size_t rowSize, int* processorChunkRows) {
    // Update intermediate rows
    for (size_t i = 1; i < nChunks; i++) {
        // Get indices
        size_t leadingEdgeRow = (size_t) processorChunkRows[i]; // 2D index
        size_t trailingEdgeRow = leadingEdgeRow - 1; // 2D index
        size_t flatLeadingEdgeRow = leadingEdgeRow * rowSize; // 1D index
        size_t flatTrailingEdgeRow = trailingEdgeRow * rowSize; // 1D index
        // Copy edge rows and columns
        cpyFlat[flatLeadingEdgeRow] = matFlat[flatLeadingEdgeRow];
        cpyFlat[flatLeadingEdgeRow + rowSize - 1] = matFlat[flatLeadingEdgeRow + rowSize - 1];
        cpyFlat[flatTrailingEdgeRow] = matFlat[flatTrailingEdgeRow];
        cpyFlat[flatTrailingEdgeRow + rowSize - 1] = matFlat[flatTrailingEdgeRow + rowSize - 1];
        // Compute values
        for (size_t j = 1; j < rowSize - 1; j++) {
            size_t flatLeadingCentre = (leadingEdgeRow * rowSize) + j; 
            size_t flatTrailingCentre = (trailingEdgeRow * rowSize) + j; 
            cpyFlat[flatLeadingEdgeRow + j] = calculateFlatNeighbourMean(matFlat, flatLeadingCentre, 4, rowSize);
            cpyFlat[flatTrailingEdgeRow + j] = calculateFlatNeighbourMean(matFlat, flatTrailingCentre, 4, rowSize);
        }
    }
}

bool flatPrecisionStopCheck(float* matFlat, float* cpyFlat, size_t size) {
    // Check if all elements are within PRECISION of previous iteration.
    for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            size_t index = (i * size) + j;
            if (fabs(matFlat[index] - cpyFlat[index]) > PRECISION) {
                return false; // At least one element in the matrix is outside PRECISION
            }
        }
    }
    return true;
}

void arrayBorderCopy(float* matFlat, float* cpyFlat, size_t n, size_t m) {
    size_t topLeft = 0;
    size_t topRight = m - 1;
    size_t bottomLeft = (n * m) - m;
    size_t bottomRight = (n * m) - 1;
    // Copy corners
    cpyFlat[topLeft] = matFlat[topLeft];
    cpyFlat[topRight] = matFlat[topRight];
    cpyFlat[bottomLeft] = matFlat[bottomLeft];
    cpyFlat[bottomRight] = matFlat[bottomRight];
    // Copy rows
    for (size_t i = 1; i < m - 1; i++) {
        cpyFlat[i] = matFlat[i];
        cpyFlat[bottomLeft + i] = matFlat[bottomLeft + i];
    }
    // Copy cols
    for (size_t i = 1; i < n - 1; i++) {
        cpyFlat[topLeft + (i * m)] = matFlat[topLeft + (i * m)];
        cpyFlat[topRight + (i * m)] = matFlat[topRight + (i * m)];
    }
}

void receiveChunkSizes(size_t* n, size_t* m) {
    // Receive chunk dimensions from master
    int sizeBuf[2];
    MPI_Recv(sizeBuf, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, 0);
    *n = (size_t) sizeBuf[0];
    *m = (size_t) sizeBuf[1];
}

void processChunk(float* matFlat, float* cpyFlat, size_t n, size_t m) {
    // Perform calcualtion of new values and write to copy.
    for (size_t i = 1; i < n - 1; i++) {
        for (size_t j = 1; j < m - 1; j++) {
            size_t centre = (i * m) + j; 
            cpyFlat[centre] = calculateFlatNeighbourMean(matFlat, centre, 4, m);
        }
    }
}

void terminateSlaves(int* processorChunkSizes, size_t nChunks, size_t size) {
    // Send negative array to slaves to terminate.
    for (size_t i = 0; i < nChunks; i++) {
        int rows = processorChunkSizes[i];
        float terminal[rows * (int) size];
        terminal[0] = (float) -1.0;
        int worker = (int) i + 1;
        MPI_Send(terminal, rows * (int) size, MPI_FLOAT, worker, 1, MPI_COMM_WORLD);
    }
}

bool terminated(float* matFlat) {
    // Check if message contains a negative value to terminate.
    return matFlat[0] < 0.0;
}

void relaxationMaster(float** mat, size_t size, size_t nProcessors, bool logging) {
    bool stop = false;
    size_t nChunks = nProcessors - 1;
    int processorChunkSizes[nChunks];
    int processorChunkRows[nChunks];
    float* matFlat = (float*) malloc(size * size * sizeof(float));
    float* cpyFlat = (float*) malloc(size * size * sizeof(float));
    flattenRows(matFlat, mat, 0, size, size); // Flatten mat.
    memcpy(cpyFlat, matFlat, size * size * sizeof(float)); // Create copy of mat.

    // Configure chunks
    calculateProcessorChunkSizes(processorChunkSizes, size, nChunks);
    calculateProcessorChunkRows(processorChunkRows, processorChunkSizes, nChunks);
    distributeChunkSizes(processorChunkRows, nChunks, size);

    // Begin relaxation process
    while (!stop) {
        if (logging) logSquareFloatMatrix(mat, size);
        distributeFlatChunks(processorChunkRows, nChunks, matFlat, size); // Distribute to worker cores
        updateFlatEdgeRows(matFlat, cpyFlat, nChunks, size, processorChunkRows); // Update edge rows in chunks that don't have 4 neighbours
        collateFlatChunks(cpyFlat, size, processorChunkRows, nChunks); // Receive completed computations from worker cores
        stop = flatPrecisionStopCheck(matFlat, cpyFlat, size);
        arraySwap(&matFlat, &cpyFlat);
        if (stop) {
            terminateSlaves(processorChunkSizes, nChunks, size); // Send termination signal to slaves.
            mat = reshapeRows(matFlat, size, size); // Reshape into 2D matrix
            if (logging) logSquareFloatMatrix(mat, size);
        }
    }
    free(matFlat);
    free(cpyFlat);
    freeFloatMatrix(mat);
}

void relaxationSlave() {
    bool stop = false;
    size_t n, m;
    receiveChunkSizes(&n, &m); // Receive dimensions of matrix chunk
    float matFlat[n * m];
    float cpyFlat[n * m];
    while (!stop) {
        MPI_Recv(&matFlat, (int) n * (int) m, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, 0);
        arrayBorderCopy(matFlat, cpyFlat, n, m); 
        if (terminated(matFlat)) { // Check if termination message
            stop = true;
        } 
        else {
            processChunk(matFlat, cpyFlat, n, m); // Perform relaxation calculations
            MPI_Send(&cpyFlat[m], (int) m * ((int) n - 2), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
        }
    }
}

void relaxation(char* dataFilePath, size_t nProcessors, int mpiRank, bool logging) {
    if (mpiRank == 0) { // master
        size_t size;
        float** mat = inputFloatMatrix(dataFilePath, &size); // Read input file
        struct timespec start, stop, delta;
        clock_gettime(CLOCK_REALTIME, &start);
        relaxationMaster(mat, size, nProcessors, logging);
        clock_gettime(CLOCK_REALTIME, &stop);
        timespecDifference(start, stop, &delta);
        float duration = floatTime(delta);
        logDuration(size, duration, nProcessors);
    }
    else { // slave
        relaxationSlave();
    }
}

int main(int argc, char** argv) {
    // Main function. Should be invoked from command line as follows:
    // ./parallel path/to/matrix/file.txt
    // An example matrix file is attached in 8.txt
    size_t dataFileArgvIndex = (size_t) argc - 1;
    char* dataFilePath = argv[dataFileArgvIndex];

    // MPI Setup
    int mpiInitReturnCode = MPI_Init(&argc, &argv);
    if (mpiInitReturnCode != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, mpiInitReturnCode);
    }

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    relaxation(dataFilePath, (size_t) mpiSize, mpiRank, LOGGING);

    MPI_Finalize();

    return 0;
}
