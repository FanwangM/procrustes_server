import numpy as np

# Create a simple 3x3 matrix
matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Save two identical copies
np.savez("matrix1.npz", arr_0=matrix)
np.savez("matrix2.npz", arr_0=matrix)
