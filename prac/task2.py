import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA



# STEP 1

# Завантаження зображення
image_raw = imread("photo.jpg")

# Виведення розмірів зображення
print("Size of first photo:", image_raw.shape)

# Відображення початкового зображення
plt.imshow(image_raw)
plt.title("First photo")
plt.axis('off')
plt.show()

# сумуємо пікселі по RGB виміру
image_sum = image_raw.sum(axis=2)

# нормалізували значення пікселів, щоби вони були в діапазоні від 0 до 1
image_bw = image_sum / image_sum.max()
# поділили на максимальну сумму по RGB та розприділили діапазон від 0 до 1


# STEP 2

# Виведення розмірів чорно-білого зображення
print("Size of black-white photo:", image_bw.shape)

# Відображення чорно-білого зображення
plt.imshow(image_bw, cmap='gray')
plt.title("")
plt.axis('off')
plt.show()


# STEP 3

# Застосування PCA
pca = PCA()
pca.fit(image_bw)

# Кумулятивна дисперсія
cumulative_dispersion = np.cumsum(pca.explained_variance_ratio_)

# Знаходження кількості компонент для покриття 95% дисперсії
number_components = np.argmax(cumulative_dispersion >= 0.95) + 1
print(f"Кількість компонентів для 95% дисперсії: {number_components}")

# Графік кумулятивної дисперсії
plt.plot(cumulative_dispersion)
plt.xlabel("Numbers of components")
plt.ylabel("Сumulative dispersion")
plt.title("Кумулятивна дисперсія залежно від кількості компонент")


# Графік кумулятивної дисперсії
plt.plot(cumulative_dispersion)
plt.xlabel("Numbers of components")
plt.ylabel("Сumulative dipersion")
plt.title("Кумулятивна дисперсія залежно від кількості компонент")
plt.axvline(number_components, color='r', linestyle='--')
plt.axhline(y=0.95, color='b', linestyle='--')
plt.show()

# STEP 4
# Реконструкція зображення з обмеженою кількістю компонент
pca = PCA(n_components=number_components)
image_bw_pca = pca.fit_transform(image_bw)
image_bw_reconstructed = pca.inverse_transform(image_bw_pca)

# Відображення реконструйованого зображення
plt.imshow(image_bw_reconstructed, cmap='gray')
plt.title(f"Реконструйоване зображення з {number_components} компонентами")
plt.axis('off')
plt.show()


# Функція для реконструкції зображення з заданою кількістю компонент
def reconstruct_image(image, n_components):
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image)
    image_reconstructed = pca.inverse_transform(image_pca)
    return image_reconstructed

components_list = [5, 20, 50, number_components, 100, 200]

# Реконструкція та відображення зображень для різної кількості компонент
plt.figure(figsize=(15, 10))
for i, n_components in enumerate(components_list):
    plt.subplot(2, 3, i+1)
    image_reconstructed = reconstruct_image(image_bw, n_components)
    plt.imshow(image_reconstructed, cmap='gray')
    plt.title(f"{n_components} компонентів")
    plt.axis('off')

plt.show()