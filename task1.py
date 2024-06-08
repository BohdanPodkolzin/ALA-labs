import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 1 =======================================================================
def find_eigen_values_and_vectors(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix should be NxN size")

    # обчислення власних значень та веторів за допомогою бібліотеки numpy
    eigenvalues, eigenvectors = np.linalg.eig(matrix)


    # для кожного власного значення, перевіряємо
    # A * v = λ * v
    for i in range(len(eigenvalues)):
        # власне значення - λ
        lambda_i = eigenvalues[i]

        # self vector v
        v_i = eigenvectors[:, i]

        # Добуток матриці на вектор A * v
        A_v = np.dot(matrix, v_i)

        # λ * v
        lambda_v = lambda_i * v_i

        # Використовуємо all close чи перевірити чи насправді вони однакові
        if not np.allclose(A_v, lambda_v):
            print(f"{A_v}, {lambda_v}\nA * v = λ * v has not accomplished")
        else:
            print(f"{A_v},\nA * v = λ * v has accomplished")

    return (eigenvalues, eigenvectors)

matrix = np.array([
                   [4, -2],
                   [1, 1]])
eigenvalues, eigenvectors = find_eigen_values_and_vectors(matrix)

print("Eigen value:", eigenvalues)
print("Eigen vectors:\n", eigenvectors)


# 2 =======================================================================


