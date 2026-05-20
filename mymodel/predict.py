import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ======================
# CONFIGURATION
# ======================

IMG_SIZE = 224
MODEL_PATH = "lung_cancer_model.keras"

# Update according to your dataset classes
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# ======================
# LOAD MODEL
# ======================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ======================
# IMAGE PREDICTION FUNCTION
# ======================

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_class_index]

    confidence = prediction[0][predicted_class_index] * 100

    # Display image
    plt.imshow(img)
    plt.axis("off")

    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")

    plt.show()

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence, "%")


# ======================
# ENTER IMAGE PATH
# ======================

image_path = input("Enter image path: ")

predict_image(image_path)