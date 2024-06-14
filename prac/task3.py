import numpy as np

# Функція для шифрування повідомлення
def encrypt_message(message, key_matrix):
    # перевели char в число по таблиці ASCII
    message_vector = np.array([ord(char) for char in message])

    # Обчислили власні значення та вектори
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    # діагонолізували матрицю
    # np.dot(eigenvectors, np.diag(eigenvalues)) - перемножили матрицю власних векторів з діагональною матрицею власних значень
    # Результат помножили на обернену матрицю власних векторів np.linalg.inv(eigenvectors), щоби знайти діагоналізовану матрицю
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

# Функція для розшифрування повідомлення
def decrypt_message(encrypted_vector, key_matrix):

    # Обчислили власні значення та вектори
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    # np.dot(eigenvectors, np.diag(1 / eigenvalues) - перемножили матрицю власних векторів з оберненою діагональною матрицею власних значень
    diagonalized_key_matrix_inv = np.dot(np.dot(eigenvectors, np.diag(1 / eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(diagonalized_key_matrix_inv, encrypted_vector)

    # перевели по ASCII табличці числа в букви та сформували розшифроване повідомлення
    decrypted_message = ''.join(chr(int(np.round(np.real(num)))) for num in decrypted_vector)
    return decrypted_message

# Генерація випадкової матриці ключа
message = "Hello, World!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))

# Шифрування повідомлення
encrypted_message = encrypt_message(message, key_matrix)
print("Encrypted Message:", encrypted_message)

# Розшифрування повідомлення
decrypted_message = decrypt_message(encrypted_message, key_matrix)
print("Decrypted Message:", decrypted_message)
