import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import random


# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# Function to create a model with random hyperparameters
def create_model(num_layers, units_per_layer, activation_function, dropout_rate):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28 * 28,)))
    
    # Adding hidden layers
    for _ in range(num_layers):
        model.add(layers.Dense(units_per_layer, activation=activation_function))
        model.add(layers.Dropout(dropout_rate))  # Optional dropout layer
    
    model.add(layers.Dense(10, activation='softmax'))  # Output layer
    return model
    # Generate random hyperparameters
num_layers = random.randint(1, 5)  # Random number of hidden layers between 1 and 5
units_per_layer = random.choice([32, 64, 128, 256])  # Random units in hidden layers
activation_function = random.choice(['relu', 'sigmoid', 'tanh'])  # Random activation function
dropout_rate = random.uniform(0.0, 0.5)  # Random dropout rate between 0.0 and 0.5

# Create the model
model = create_model(num_layers, units_per_layer, activation_function, dropout_rate)
# Create the model
model = create_model(num_layers, units_per_layer, activation_function, dropout_rate)

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'\nTest accuracy: {test_acc:.4f}')
