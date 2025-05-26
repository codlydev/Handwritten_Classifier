# handwritten_digit_classifier.py

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and Preprocess the Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 2: Visualize Sample Images
plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 3: Build the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Flatten input
    layers.Dense(128, activation='relu'),          # Hidden layer
    layers.Dropout(0.2),                           # Dropout to prevent overfitting
    layers.Dense(10, activation='softmax')         # Output layer (10 classes)
])

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Step 7: Make Predictions on Test Data
predictions = model.predict(x_test)

# Step 8: Display One Prediction
index = 0  # change index to test other digits
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Prediction: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
plt.axis('off')
plt.show()

# Step 9: Save the Trained Model
model.save("mnist_digit_classifier.h5")
print("\n📦 Model saved as mnist_digit_classifier.h5")
