import time
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# def main():
#     print("Starting Hyperparameter Estimation process...")
#     time.sleep(5)  # Simulating some work
#     print("Hyperparameter Estimation completed.")
#     print("Best hyperparameters have been saved to the database.")

def load_data():
    # Load the MNIST dataset
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the data
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape data to fit the model input
    x_train_full = x_train_full.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # Split the full training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, 
        y_train_full, 
        test_size=0.1, 
        random_state=42
    )

    print(type(x_test))
    return x_train, x_val, y_train, y_val

    
    
def test_hyperparameter_set(
    layer_depth,
    layer_size,
    x_train,
    y_train,
    x_val,
    y_val
):
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(28 * 28,)))  # Input layer

    # Add hidden layers
    for _ in range(layer_depth):
        model.add(layers.Dense(layer_size, activation='relu'))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=5,  # You can adjust the number of epochs
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=0  # Set to 1 to see training progress
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    print(f"Depth: {layer_depth}, Size: {layer_size}, Val Accuracy: {val_accuracy:.4f}")
    
    hyperparameter_dict = {
        'Depth': layer_depth,
        'Size': layer_size,
        'Val Accuracy': val_accuracy
    }
    return hyperparameter_dict
    
def search_hyperparameter():
    x_train, x_val, y_train, y_val = load_data()
    
    # Define the range for layer depth and layer size
    layer_depths = [1, 2, 3]  # Number of hidden layers
    layer_sizes = [32, 64, 128]  # Number of neurons in each hidden layer

    best_accuracy = 0.0
    best_params = {}
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
                    y_val
                
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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data_{timestamp}.json"

    # Save the dictionary as a JSON file
    with open(filename, 'w') as file:
        json.dump(max_dict, file, indent=4)

    print(f"Data saved to {filename}")

            
    

if __name__ == "__main__":
    # main()
    search_hyperparameter()