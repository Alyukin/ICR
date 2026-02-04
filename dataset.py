import tensorflow as tf
from tensorflow.keras import layers, models
# import numpy as np
# import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Пиксели - 0-255. Приводим к диапазону 0-1 для обучения.
x_train, x_test = x_train / 255.0, x_test / 255.0

# reshape  (count, height, width, channels)
# изображение черно-белое -> канал = 1
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

print(f"Размер тренировочных данных: {x_train.shape}")


model = models.Sequential([
    # выделение признаков по слоям
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # полносвязные слои (классификация)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 выходов = цифры 0-9
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Begin model train")

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.save('handwritten_digit_recognition.h5')
print("Model saved 'handwritten_digit_recognition.h5'")