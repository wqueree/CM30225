import numpy as np

for exp in range(2, 12):
    size = 2 ** exp
    with open(file=f"{size}.txt", mode="w") as f:
        f.write(f"{size}\n")
        for _ in range(size):
            for _ in range(size - 1):
                f.write(f"{np.round(np.random.uniform(), 2)} ")
            f.write(f"{np.round(np.random.uniform(), 2)}\n")