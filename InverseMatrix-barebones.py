# This file is the python template for the 1st assignment of Calculus
# and Linear Algebra. Feel free to adapt it to your needs.
import numpy as np


# Function to obtain the index of the row with the maximum absolute
# value (in the column col
def index_of_max_abs(A, row, col):
    idx_max = row  # Starting from first index
    for i in range(row + 1, A.shape[0]):  # Check all rows in the column
        if abs(A[i, col]) > abs(A[idx_max, col]):  # If it is larger, then change index max
            idx_max = i
    return idx_max


# Function to swap the rows row1 and row2 in the matrix A. Since the
# matrix is passed by reference the it will be changed after calling
# the function, i.e.  no need to return a new matrix
def swap_rows(A, row1, row2):
    for i in range(0, A.shape[1]):  # From 0 since matrix in python is 0-indexed
        temp = A[row1, i]  # Store in temp variable
        A[row1, i] = A[row2, i]  # Swap
        A[row2, i] = temp  # Swap


def gauss_elimination(A, m, n):  # Making the matrix in upper triangular form
    pivot_row = 0  # Matrix in python is 0-indexed
    pivot_col = 0  # Matrix in python is 0-indexed

    while pivot_row <= m - 1 and pivot_col <= n - 1:  # Subtracting 1 because of 0-indexed matrix
        idx_max = index_of_max_abs(A, pivot_row, pivot_col)
        if abs(A[idx_max, pivot_col]) < 0.0001:  # 'Column' of zeros. Check if close to 0 (numerical errors)
            pivot_col += 1  # Go to next pivot, skip column
        else:
            swap_rows(A, pivot_row, idx_max)
            for i in range(pivot_row + 1, m):  # Since m is not included, we should NOT subtract 1 here
                alpha = (A[i, pivot_col]) / (A[pivot_row, pivot_col])
                A[i, pivot_col] = 0  # Avoid numerical errors
                for j in range(pivot_col + 1, n):  # Since n is not included, we should NOT subtract 1 here
                    A[i, j] -= alpha * A[pivot_row, j]
            pivot_row += 1  # Next pivot
            pivot_col += 1  # Next pivot


# Finishing making the inverse by using the upper triangular form and continue operating on it
def gauss_jordan(A, m, n):  # Matrix must be in upper triangular form first, which is done by gauss_elimination
    pivot = m - 1  # Initial pivot
    while pivot >= 0:  # 0 since matrix is 0-indexed
        if abs(A[pivot, pivot]) < 0.0001:  # Check if the value is very close to zero, and thereby 0
            raise Exception("Matrix is singular")  # Checking for near 0 is done, because computers don't calculate well
        else:
            for i in range(pivot - 1, -1, -1):  # Going down to -1 since it is not included, and matrix is 0-indexed.
                # Last -1 makes it decrement. Going from pivot-1 to -1 (not included)
                alpha = (A[i, pivot]) / (A[pivot, pivot])
                A[i, pivot] = 0  # Avoid numerical errors
                for j in range(pivot + 1, n):  # Operate whole row
                    A[i, j] -= alpha * A[pivot, j]
            alpha = 1 / A[pivot, pivot]
            for j in range(0, n):  # Starting from zero since matrix is 0-indexed. Scale row for identity
                A[pivot, j] *= alpha
            pivot -= 1  # Next pivot


# Implementation of the Gauss-Jordan elimination algorithm for inverse calculation
def inverse(A):
    # Make sure the matrix given as an argument is a square matrix
    m = A.shape[0]  # Number of rows in matrix A
    if A.shape[1] != m:
        raise Exception("Matrix must be square")

    # Create empty tildeA matrix adding the identity matrix to A
    tildeA = np.append(A, np.eye(m, dtype=np.float32), axis=1)

    # Number of columns of matrix tildeA (it should be 2*n)
    n = tildeA.shape[1]

    gauss_elimination(tildeA, m, n)  # Upper triangular form

    gauss_jordan(tildeA, m, n)  # Using Gauss-Jordan to finish the inverse

    # Extract the inverse matrix from the matrix tildeA
    InvA = tildeA[:, m:]

    InvA = np.around(InvA, 3)  # Round to three decimals

    # Return the inverse matrix
    return InvA


def test():
    # Test for the inverse matrix calculation
    # test = 0: sample 3x3 matrix
    # test = 1: sample 4x4 matrix
    # test = 2: sample 5x5 matrix
    # test = 3: random nxn matrix
    test = 0
    n = 6
    if test == 0:
        A = np.array([[0, 2, 0], [1, 4, 2], [4, 2, 0]], dtype=np.float32)
    elif test == 1:
        A = np.array([[0, 2, 0, 8], [1, 4, 2, -2], [4, 2, 0, -3], [2, -1, -6, 5]], dtype=np.float32)
    elif test == 2:
        A = np.array([[0, 2, 0, 8, 1], [1, 2, 2, -2, 5], [4, 2, 0, -3, 2], [2, -1, -6, 5, 0], [-5, 3, 8, 4, 4]],
                     dtype=np.float32)
    else:
        A = np.random.rand(n, n)

    # Calculate the inverse matrix of A
    try:  # Adding try to catch exception
        InvA = inverse(A)
        print('The inverse of matrix A:')
        print(A)
        print('Is:')
        print(InvA)

        print('The (rounded) product of Inv(A)*A is:')
        print(np.rint(np.matmul(A, InvA)))  # rint() rounds the result to the nearest integer
        print('The product of Inv(A)*A is:')
        print(np.matmul(A, InvA))  # rint() rounds the result to the nearest integer
    except Exception as e:  # Catching the exception and printing it correctly.
        print("ERROR: " + str(e))


if __name__ == "__main__":
    print('Running test')
    test()
