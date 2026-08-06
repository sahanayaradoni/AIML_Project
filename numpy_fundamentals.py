"""
numpy_fundamentals.py
---------------------
Demonstrates NumPy fundamentals:

- 1D, 2D, 3D array creation and shape verification
- Broadcasting
- Vectorised operations
- Matrix multiplication
- Mean, standard deviation, and correlation using CSV data
"""

import os
import numpy as np
import pandas as pd


def show_array(name, arr):
    """Display array and its shape."""
    print(f"\n{name}:")
    print(arr)
    print("Shape:", arr.shape)


def main():

    # 1D Array
    arr1 = np.array([10, 20, 30, 40, 50])
    assert arr1.shape == (5,)
    show_array("1D Array", arr1)


    # 2D Array
    arr2 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert arr2.shape == (2, 3)
    show_array("2D Array", arr2)


    # 3D Array
    arr3 = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    assert arr3.shape == (2, 2, 2)
    show_array("3D Array", arr3)


    # Broadcasting
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    vector = np.array([10, 20, 30])

    broadcast_result = matrix + vector

    assert broadcast_result.shape == matrix.shape

    show_array("Broadcasting Result", broadcast_result)


    # Vectorised Operations
    numbers = np.array([1, 2, 3, 4, 5])

    show_array("Square", numbers ** 2)
    show_array("Multiply by 10", numbers * 10)
    show_array("Add 5", numbers + 5)


    # Matrix Multiplication
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [5, 6],
        [7, 8]
    ])

    result = A @ B

    print("\nMatrix Multiplication:")
    print(result)


    # CSV Statistics
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "students.csv"
    )

    data = pd.read_csv(csv_path)

    numpy_data = data.to_numpy()

    print("\nMean:")
    print(np.mean(numpy_data, axis=0))

    print("\nStandard Deviation:")
    print(np.std(numpy_data, axis=0))

    print("\nCorrelation:")
    print(np.corrcoef(numpy_data, rowvar=False))


if __name__ == "__main__":
    main()