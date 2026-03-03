import math 
import numpy as np  


def matrix_multiply(matrix_a, matrix_b, transpose_a=False, transpose_b=False):
    if transpose_a:
        matrix_a = np.transpose(matrix_a)
    if transpose_b:
        matrix_b = np.transpose(matrix_b)

    # Validate dimensions
    b_rows = len(matrix_b)
    a_col = len(matrix_a[0])

    if a_col != b_rows:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
        


