import numpy as np

# Matrix dimensions
m = 1024  # rows
n = 1024   # cols 
r = 1    # rank

# Generate random low rank factors
A = np.random.rand(m, r)  # m x r matrix
B = np.random.rand(n, r)  # n x r matrix

# Construct low rank matrix
low_rank_matrix = A @ B.T  # m x n matrix

# Compute QR decomposition
Q, R = np.linalg.qr(low_rank_matrix)

print(f"Matrix dimensions: {low_rank_matrix.shape}")
print(f"Q dimensions: {Q.shape}")
print(f"R dimensions: {R.shape}")

# Construct Q[:, :r]
Q_r = Q[:, :r]
R_r = R[:r, :]

# Verify reconstruction
reconstruction = Q @ R
reconstruction_2 = Q_r @ Q_r.T @ Q @ R
reconstruction_3 = Q_r @ R_r
# error = np.max(np.abs(low_rank_matrix - reconstruction))
# error_2 = np.max(np.abs(low_rank_matrix - reconstruction_2))
# error_3 = np.max(np.abs(low_rank_matrix - reconstruction_3))
error = np.linalg.norm(low_rank_matrix - reconstruction) / n;
error_2 = np.linalg.norm(low_rank_matrix - reconstruction_2) / n;
error_3 = np.linalg.norm(low_rank_matrix - reconstruction_3) / n;
print(f"\nMax reconstruction error: {error}")
print(f"\nMax reconstruction error 2: {error_2}")
print(f"\nMax reconstruction error 3: {error_3}")

truncation = Q_r @ Q_r.T @ Q
# print(truncation.shape)
# print(Q_r.T @ Q)
