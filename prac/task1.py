## SVD realization
import numpy as np

def svd(matrix):

    # 1) Find symmetric marix 2x2 and 3x3
    # (2x2 - U matrix, 3x3T - VT matrix)

    # MATRIX 2x2
    mul_MT_M = np.dot(matrix.T, matrix)

    #matrix 3x3
    mul_M_MT = np.dot(matrix, matrix.T)

    # 2)
    # Find Eigen Values and Vectors
    eigenval_MT_M, eigenvec_MT_M = np.linalg.eigh(mul_MT_M)
    eigenval_M_MT, eigenvec_M_MT = np.linalg.eigh(mul_M_MT)

    # sort it in descending order
    desc_ind_MT_M = np.argsort(eigenval_MT_M)[::-1]
    desc_ind_M_MT = np.argsort(eigenval_M_MT)[::-1]

    eigenval_MT_M = eigenval_MT_M[desc_ind_MT_M]
    eigenval_M_MT = eigenval_M_MT[desc_ind_M_MT]

    eigenvec_MT_M = eigenvec_MT_M[:, desc_ind_MT_M]
    eigenvec_M_MT = eigenvec_M_MT[:, desc_ind_M_MT]


    # 3) Sigma Matrix

    sigma = np.sqrt(eigenval_MT_M)
    sigma_matrix = np.zeros_like(matrix, dtype=float)
    np.fill_diagonal(sigma_matrix, sigma)

    reconstructed_matrix = np.dot(eigenvec_M_MT, np.dot(sigma_matrix, eigenvec_MT_M.T))
    is_reconstructed = np.allclose(matrix, reconstructed_matrix)

    return eigenvec_MT_M, sigma_matrix, eigenvec_M_MT, is_reconstructed



matrix = np.array([
    [3, 2],
    [2, 3],
    [2, -2]
])

result = svd(matrix)
for el in result:
    print(el)
    print('\n\n')