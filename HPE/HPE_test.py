import time
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from mongodb_function import select_collection, load_image_data, insert_hyperparameter_json

def load_data():
    """
    Loading the MNIST train dataset from MongoDB.
    Reshaping the images into two dimensions and splitting it further
    into a validation dataset for the hyperparameter search

    Returns:
        x_train (numpy.ndarray): images for training
        x_val (numpy.ndarray): images for validation of results
        y_train (numpy.ndarray): labels for training
        y_val (numpy.ndarray): labels for validation of results
    """    
    train_collection = select_collection('mnist_db', 'mnist_train')
    x_train_full, y_train_full = load_image_data(train_collection)

    x_train_full = x_train_full.astype('float32') / 255.0
    x_train_full = x_train_full.reshape(-1, 28 * 28)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, 
        y_train_full, 
        test_size=0.1, 
        random_state=42
    )

    return x_train, x_val, y_train, y_val

    
def test_hyperparameter_set(
    layer_depth,
    layer_size,
    x_train,
    y_train,
    x_val,
    y_val,
    activation
):
    """
    Tests given hyperparameters for a neural network and returns a dictionary
    of hyperparameters and the accuracy.

    Args:
        layer_depth (int): The number of hidden layers in the neural network.
        layer_size (int): Size of a single layer in the network
        x_train (numpy.ndarray): Training data images.
        y_train (numpy.ndarray): Training data labels.
        x_val (numpy.ndarray): Validation data images.
        y_val (numpy.ndarray): Validation data labels.
        activation (str): activation function for the hidden layers

    Returns:
        dict: dictionary with the best hyperparameters according to accuracy
    """    
    model = keras.Sequential()
    model.add(keras.Input(shape=(28 * 28,)))  # Input layer

    for _ in range(layer_depth):
        model.add(keras.layers.Dense(layer_size, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=0
    )

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    print(f"Depth: {layer_depth}, Size: {layer_size}, Val Accuracy: {val_accuracy:.4f}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    hyperparameter_dict = {
        'Depth': layer_depth,
        'Size': layer_size,
        'Val Accuracy': val_accuracy,
        'Activation': activation,
        'Time': timestamp
    }
    return hyperparameter_dict
    
    
def search_hyperparameters(activation_function):
    """
    Performs a hyperparameter search for a neural network using the specified activation function.

    Trains models with different layer depths and sizes, evaluates their validation accuracy, and
    saves the best model's hyperparameters and accuracy to MongoDB.

    Args:
        activation_function (str): The activation function to use in the hidden layers

    Returns:
        None
    """
    x_train, x_val, y_train, y_val = load_data()
    
    layer_depths = [1, 3, 5]
    layer_sizes = [8, 32, 128]
    
    param_dicts = []
    for depth in layer_depths:
        for size in layer_sizes:
            test_hyperparameter_dict = (
                test_hyperparameter_set(
                    depth,
                    size,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    activation_function
                )
            )
            param_dicts.append(
                test_hyperparameter_dict
            )
            
    max_dict = max(
        param_dicts,
        key=lambda x: x['Val Accuracy']
    )
    print(max_dict)
    
    insert_hyperparameter_json(
        max_dict,
        activation_function
    )
    print(f"Data saved to MongoDB")
    

if __name__ == "__main__":
    search_hyperparameters('sigmoid')
    search_hyperparameters('relu')