from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1 ==============================================================
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=50, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

# # amend NaN value by avg value for each film
ratings_matrix_filled = ratings_matrix.apply(lambda col: round(col.fillna(col.mean()), 2), axis=0)

# Обчислення середнього рейтингу кожного користувача
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=2)

# 2 ======================================================
#  Чим меньша k, тим менш точно відбувається відновлення, але швидкість обчислення виростає

def svd_reconstruction(R_demeaned, key, user_ratings_mean):
    U, S, Vt = svds(R_demeaned, k=key)
    S = np.diag(S)
    reconstructed_R = np.dot(U, np.dot(S, Vt)) + user_ratings_mean.reshape(-1, 1)

    return reconstructed_R

keys = [2, 3, 4, 10]
reconstructions = []

for k in keys:
    reconstructed_R = svd_reconstruction(R_demeaned, k, user_ratings_mean)
    # Для кожного значення
    # k відновлюємо матрицю та обчислюємо середньоквадратичну помилку (MSE)
    error = np.mean((R - reconstructed_R) ** 2)
    reconstructions.append(reconstructed_R)

plt.figure(figsize=(10, 6))
plt.plot(keys, marker='o')
plt.title('Reconstruction')
plt.xlabel('Number of Singular Values (k)')
plt.grid(True)
plt.show()
