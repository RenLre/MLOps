import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mongodb')))
from mongodb_function import load_image_data, load_hyperparameters, select_collection
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

def load_data():
    # Load the MNIST dataset
    train_collection = select_collection('mnist_db', 'mnist_train')
    x_train, y_train = load_image_data(train_collection)
    
    test_collection = select_collection('mnist_db', 'mnist_test')
    x_test, y_test = load_image_data(test_collection)

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape data to fit the model input
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    return x_train, x_test, y_train, y_test


def create_model(num_layers, units_per_layer, activation_function):
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(28 * 28,)))  # Input layer

    # Add hidden layers
    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(units_per_layer, activation=activation_function)
            )

    # Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model


def train_and_evaluate_model(
    hyperparameters,
    x_train,
    y_train,
    x_val,
    y_val
):
    
    # Use the hyperparameters loaded from MongoDB
    model = create_model(
        num_layers=hyperparameters['Depth'],  # From MongoDB
        units_per_layer=hyperparameters['Size'],  # From MongoDB
        activation_function=hyperparameters['Activation'],
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_val, y_val)
    print(f'\nTest accuracy: {test_acc:.4f}')

    return model


def pickle_model(model):
    # Save the model to a .pickle file
    model_filename = 'trained_model.pickle'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    return model_filename

def upload_model_to_mongo(model_filename):
    collection = select_collection('model_db', 'model_collection')

    # Read the .pickle file and store as binary
    with open(model_filename, 'rb') as f:
        model_data = f.read()

    # Insert the model data into MongoDB
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_document = {
        'model_name': model_filename,
        'model_data': model_data,
        'timestamp': timestamp
    }
    
    collection.insert_one(model_document)
    print(f'Model {model_filename} uploaded to MongoDB successfully.')

def final_prediction():
    relu_hyperparameters =  load_hyperparameters('relu')
    sigmoid_hyperparameters =  load_hyperparameters('sigmoid')
    x_train, x_val, y_train, y_val = load_data()
    sigmoid_better = (
        sigmoid_hyperparameters['Val Accuracy']
        > relu_hyperparameters['Val Accuracy']
    )
    if sigmoid_better:
        hyperparameter_dict = sigmoid_hyperparameters
    else:
        hyperparameter_dict = relu_hyperparameters
    
    model = train_and_evaluate_model(
        hyperparameters=hyperparameter_dict,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )
    
    pickle_path = pickle_model(model)
    upload_model_to_mongo(pickle_path)
    
    
if __name__ == "__main__":
    final_prediction()
        
    