#include <stdio.h>

int main() {
    int x = 8;
    int y = x;
    x = x + 1;

    printf("%p: %d\n%p: %d\n", &x, x, &y, y);
    return 0;
}