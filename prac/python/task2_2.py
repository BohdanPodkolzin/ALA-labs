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

# amend NaN value by avg value for each film
ratings_matrix_filled = ratings_matrix.apply(lambda col: round(col.fillna(col.mean()), 2), axis=0)

# Обчислення середнього рейтингу кожного користувача
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# 2 ======================================================

U, sigma, Vt = svds(R_demeaned, k=2)


# tests =====================================
# Де k відповідає за розмірність даних, яку зберігаємо
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
    # k відновлюємо матрицю та обчислюємо середньоквадратичну помилкy
    error = np.mean((R - reconstructed_R) ** 2)
    reconstructions.append(error)

plt.figure(figsize=(10, 6))
plt.plot(keys, reconstructions, marker='o')
plt.title('Reconstruction errors and Number of singular val. Graph')
plt.xlabel('Number of singular values')
plt.ylabel('Mean squared error')
plt.grid(True)
plt.show()
#  Чим меньша k, тим менш точно відбувається відновлення, але швидкість обчислення виростає
# Отже, оскліьки зростає кількість помилок, значить зростає похибка самого підбору схожих данних для юзерів
# Це можна використовувати аби юзеру рукомендувало не однотипні серіали на базі того зо він подивився, а разброс був більший


# 3 ===================================

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

# 5 =======================================

# print(R_demeaned)
# print(all_user_predicted_ratings)
# all_user_predicted_ratings будуть близькими до початкових, але можуть мати деякі відхилення залежно від значення
# Менше значення k може призвести до більш значних відхилень.

# 6

predicted_only = ratings_matrix.copy()

# Заміна початкових оцінок на NaN
predicted_only[~ratings_matrix.isna()] = np.nan

# Вставка спрогнозованих оцінок на місце NaN
predicted_only.fillna(round(preds_df, 2), inplace=True)

# print("Таблиця з прогнозованими оцінками:")
# print(predicted_only)


# 8 ==================================


df_movies = pd.read_csv('movies.csv')

def recommend_movies(user_id, preds_df, movies_df, ratings_matrix, num_recommendations=10):
    user_pred_ratings = preds_df.loc[user_id]

    # Викидуємо оцінені юзером фільми
    user_rated_movies = ratings_matrix.loc[user_id].dropna().index
    user_pred_ratings = user_pred_ratings.drop(user_rated_movies, errors='ignore').sort_values(ascending=False)

    # Схрещуємо інфу про фільми та предіктабл рейтинг
    recommendations = (movies_df.set_index('movieId')
                       .loc[user_pred_ratings.index]
                       .assign(predicted_rating=user_pred_ratings.values))

    # Берем топ 10 рекомендацій для юзера
    top_recommendations = recommendations.head(num_recommendations)

    return top_recommendations[['title', 'genres', 'predicted_rating']]

user_id = 200
top_10_movies = recommend_movies(user_id, preds_df, df_movies, ratings_matrix)
print(top_10_movies)