import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def create_model(num_layers, units_per_layer, activation_function, dropout_rate):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28 * 28,)))
    for _ in range(num_layers):
        model.add(layers.Dense(units_per_layer, activation=activation_function))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def generate_hyperparameters():
    return {
        'num_layers': random.randint(1, 5),
        'units_per_layer': random.choice([32, 64, 128, 256]),
        'activation_function': random.choice(['relu', 'sigmoid', 'tanh']),
        'dropout_rate': random.uniform(0.0, 0.5)
    }

def train_and_evaluate_model():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    hyperparameters = generate_hyperparameters()
    model = create_model(**hyperparameters)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')
    return test_acc

# This function will be called by your Airflow task
def run_model_training():
    return train_and_evaluate_model()

# Only execute if this script is run directly (not imported)
if __name__ == "__main__":
    run_model_training()