import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models
import tensorflow as tf
# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))


# Pad or truncate sequences to ensure all have the same length
data = np.asarray(data_dict['data'])
labels = np.array(data_dict['labels'])
labels = tf.keras.utils.to_categorical(labels)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(42,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(15, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")

# Save the model architecture as an image
model.save(
    r"D:\sign langauge\American-Sign-Language-Alphabet-Recognition-by-mediapipe-and-ML-computer-vision-main\American-Sign-Language-Alphabet-Recognition-by-mediapipe-and-ML-computer-vision-main\result\model.h5")

