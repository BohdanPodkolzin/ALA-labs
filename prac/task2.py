from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1, 2, 3, 4, 5, 6, 7 ===========================

# Зчитування CSV файлу
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

# amend NaN value by avg value for each film
ratings_matrix_filled = ratings_matrix.apply(lambda col: round(col.fillna(col.mean()), 2), axis=0)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

# 9 =====================================
U_ten_users = U[:10]

# Створення тривимірного графіка
U_figure = plt.figure()
axis = U_figure.add_subplot(111, projection='3d')

for i in range(10):
    axis.scatter(U_ten_users[i, 0], U_ten_users[i, 1], U_ten_users[i, 2])

plt.title('Ratings in 3D Space')
plt.show()

# 11, 13 =====================

print(Vt)

Vt_fig = plt.figure()
axis = Vt_fig.add_subplot(111, projection='3d')

for i in range(3):
    axis.scatter(Vt[i, 0], Vt[i, 1], Vt[i, 2])

plt.title('Films in 3D Space')
plt.show()



# ======================================= 2_2 ===============================





