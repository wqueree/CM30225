import os
import filecmp
import subprocess
import numpy as np

from typing import Iterable
from pathlib import Path

PRECISION = 0.01


def write_square(size: int) -> Path:
    filename = f"{size}.txt"
    with open(file=filename, mode="w") as f:
        f.write(f"{size}\n")
        for _ in range(size):
            for _ in range(size - 1):
                f.write(f"{np.round(np.random.uniform(), 2)} ")
            f.write(f"{np.round(np.random.uniform(), 2)}\n")
    return Path(filename)


def precision_equals(
    parallel_result: Iterable[float], serial_result: Iterable[float], precision: float
) -> bool:
    for (parallel_element, serial_element) in zip(parallel_result, serial_result):
        if abs(parallel_element - serial_element) > precision:
            return False
    return True


def relaxation_compile() -> None:
    subprocess.run(
        ["gcc", "-fdiagnostics-color=always", "-g", "../serial.c", "-o", "../serial"]
    )
    subprocess.run(
        [
            "mpicc",
            "-fdiagnostics-color=always",
            "-g",
            "../parallel.c",
            "-o",
            "../parallel",
        ]
    )


def relaxation_test(test_data_filename: Path, precision: float) -> bool:
    serial_result_path = f"serial-{test_data_filename.stem}.out"
    parallel_result_path = f"parallel-{test_data_filename.stem}.out"

    # Compute Results
    with open(serial_result_path, "w") as serial_result_file:
        subprocess.call(
            ["../serial", str(test_data_filename)], stdout=serial_result_file
        )
    with open(parallel_result_path, "w") as parallel_result_file:
        subprocess.call(
            ["mpirun", "-np", "4", "../parallel", str(test_data_filename)],
            stdout=parallel_result_file,
        )

    # Check Results
    with open(serial_result_path, "r") as serial_result_file:
        serial_result = map(
            float, serial_result_file.read().replace("\n", "").strip().split(" ")
        )
    with open(parallel_result_path, "r") as parallel_result_file:
        parallel_result = map(
            float, parallel_result_file.read().replace("\n", "").strip().split(" ")
        )

    equal = precision_equals(parallel_result, serial_result, precision)
    os.remove(serial_result_path)
    os.remove(parallel_result_path)
    os.remove(test_data_filename)
    return equal


def test_correctness(runs: int, precision: float) -> None:
    relaxation_compile()
    for i in range(runs):
        n = np.random.randint(20, 100)
        test_data_filename = write_square(n)
        print(
            f"Test {i + 1} ({n=}): {'PASS' if relaxation_test(test_data_filename, precision) else 'FAIL'}"
        )


test_correctness(runs=40, precision=PRECISION)
