import numpy as np
from scipy import sparse

n = 16  # Change this value to adjust the size of the array

sparse_result = sparse.eye(n, dtype=int, format="lil")

mask = np.zeros((n,), dtype=bool)
mask[0::3] = True
sparse_result[mask, mask] = 0

# Convert the modified sparse matrix to a dense NumPy array
result = sparse_result.toarray()

print(result)
