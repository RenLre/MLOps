import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from pymongo import MongoClient
import pickle
import os

# Load the hyperparameters from MongoDB
def load_hyperparameters_from_mongo(collection_name):
    # Connect to MongoDB 
    client = MongoClient("mongodb://localhost:27017/")  
    db = client['your_database_name']  
    collection = db[collection_name]

    # Fetch the latest hyperparameter set 
    hyperparameters = collection.find_one(sort=[("timestamp", -1)])  # timestamp field 

    if hyperparameters:
        # Extract relevant fields; adjust based on your MongoDB schema
        return {
            'Depth': hyperparameters.get('Depth'),
            'Size': hyperparameters.get('Size'),
            # Include any other fields you want to load from MongoDB
        }
    else:
        print("No hyperparameters found in MongoDB.")
        return None

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

def train_and_evaluate_model(hyperparameters):
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Use the hyperparameters loaded from MongoDB
    model = create_model(
        num_layers=hyperparameters['Depth'],  # From MongoDB
        units_per_layer=hyperparameters['Size'],  # From MongoDB
        activation_function='relu',  # Set a default, or modify to read from MongoDB if needed
        dropout_rate=0.0  # Adjust as needed
    )
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    # Save the model to a .pickle file
    model_filename = 'trained_model.pickle'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Upload the model to MongoDB
    upload_model_to_mongo(model_filename)

    return test_acc

def upload_model_to_mongo(model_filename):
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client['your_database_name']
    collection = db['models']  # Change to your desired collection name

    # Read the .pickle file and store as binary
    with open(model_filename, 'rb') as f:
        model_data = f.read()

    # Insert the model data into MongoDB
    model_document = {
        'model_name': model_filename,
        'model_data': model_data,
        'timestamp': datetime.now()  # Optional timestamp for reference
    }
    
    collection.insert_one(model_document)
    print(f'Model {model_filename} uploaded to MongoDB successfully.')

# This function will be called by your Airflow task
def run_model_training(collection_name):
    hyperparameters = load_hyperparameters_from_mongo(collection_name)
    if hyperparameters:
        return train_and_evaluate_model(hyperparameters)
    else:
        print("Error: No hyperparameters found in MongoDB.")
        return None

# Only execute if this script is run directly (not imported)
if __name__ == "__main__":
    collection_name = "your_collection_name"  # Replace with your MongoDB collection name
    run_model_training(collection_name)
