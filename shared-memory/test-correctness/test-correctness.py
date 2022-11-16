import filecmp
import subprocess
import numpy as np

from pathlib import Path

def write_square(size: int) -> Path:
    filename = f"{size}.txt"
    with open(file=filename, mode="w") as f:
        f.write(f"{size}\n")
        for _ in range(size):
            for _ in range(size - 1):
                f.write(f"{np.round(np.random.uniform(), 2)} ")
            f.write(f"{np.round(np.random.uniform(), 2)}\n")
    return Path(filename)


def relaxation_compile() -> None:
    subprocess.run(["gcc", "-fdiagnostics-color=always", "-g", "../serial.c", "-o", "../serial", "-lpthread"])
    subprocess.run(["gcc", "-fdiagnostics-color=always", "-g", "../parallel.c", "-o", "../parallel", "-lpthread"])

def relaxation_test(test_data_filename: Path):
    serial_result_path = f"serial-{test_data_filename.stem}.out"
    parallel_result_path = f"parallel-{test_data_filename.stem}.out"
    with open(serial_result_path, "w") as serial_result:
        subprocess.call(["../serial", str(test_data_filename)], stdout=serial_result)
    with open(parallel_result_path, "w") as parallel_result:
        subprocess.call(["../parallel", str(test_data_filename)], stdout=parallel_result)
    return filecmp.cmp(serial_result_path, parallel_result_path, shallow=False)

def test_correctness(runs: int):
    results = []
    relaxation_compile()
    for _ in range(runs):
        test_data_filename = write_square(np.random.randint(4, 33))
        results.append(relaxation_test(test_data_filename))
    return results

print(test_correctness(runs=1))