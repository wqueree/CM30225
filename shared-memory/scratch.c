#include <stdio.h>

void readDataFile(double** mat, FILE* data, size_t size) {
    
}

int main(int argc, char **argv) {
    FILE* fr = fopen(argv[1], "r");
    long size = 0;
    fscanf(fr, "%ld,", &size);
    printf("%ld\n", size);
    fclose(fr);
    return 0;
}