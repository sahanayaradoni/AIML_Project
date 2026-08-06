import numpy as np
import pandas as pd

# 1D Array
arr1 = np.array([10, 20, 30, 40, 50])

print("1D Array:")
print(arr1)
print("Shape:", arr1.shape)


# 2D Array
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n2D Array:")
print(arr2)
print("Shape:", arr2.shape)


# 3D Array
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("\n3D Array:")
print(arr3)
print("Shape:", arr3.shape)


# Broadcasting
print("\nBroadcasting:")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

vector = np.array([10, 20, 30])

print(matrix + vector)


# Vectorised Operations
print("\nVectorised Operations:")

numbers = np.array([1, 2, 3, 4, 5])

print("Square:", numbers ** 2)
print("Multiply by 10:", numbers * 10)
print("Add 5:", numbers + 5)


# Matrix Multiplication
print("\nMatrix Multiplication:")

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)


# CSV Statistics

data = pd.read_csv("students.csv")

numpy_data = data.to_numpy()

print("\nMean:")
print(np.mean(numpy_data, axis=0))

print("\nStandard Deviation:")
print(np.std(numpy_data, axis=0))

print("\nCorrelation:")
print(np.corrcoef(numpy_data, rowvar=False))