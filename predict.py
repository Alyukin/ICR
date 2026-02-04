import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_digit(img_path):
    try:
        model = tf.keras.models.load_model('handwritten_digit_recognition.h5')
    except:
        print("Err: Model is not loading properly")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Err: Cannot open {img_path}.")
        return

    img = cv2.bitwise_not(img)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_norm = img_resized / 255.0
    input_data = img_norm.reshape(1, 28, 28, 1)

    prediction = model.predict(input_data)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"----------------------")
    print(f"digit: {digit}")
    print(f"confidence: {confidence:.2f}%")
    print(f"----------------------")


    plt.imshow(img_resized, cmap='gray')
    plt.title(f"prediction: {digit} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

predict_digit('./test_img/digit.png')