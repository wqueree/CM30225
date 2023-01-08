import numpy as np

for size in range(8, 17, 20):
    with open(file=f"{size}.txt", mode="w") as f:
        f.write(f"{size}\n")
        for _ in range(size):
            for _ in range(size - 1):
                f.write(f"{np.round(np.random.uniform(), 2)} ")
            f.write(f"{np.round(np.random.uniform(), 2)}\n")