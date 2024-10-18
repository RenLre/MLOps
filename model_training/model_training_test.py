import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mongodb')))
from mongodb_function import load_image_data, load_hyperparameters, select_collection
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

def load_data():
    """
    Loads and preprocesses the MNIST dataset from MongoDB.
    The dataset is already split into test and train set

    The data is normalized and reshaped.

    Returns:
        tuple: A tuple containing:
            x_train (numpy.ndarray): Training images.
            x_test (numpy.ndarray): Testing images.
            y_train (numpy.ndarray): Training labels.
            y_test (numpy.ndarray): Testing labels.
    """
    train_collection = select_collection('mnist_db', 'mnist_train')
    x_train, y_train = load_image_data(train_collection)
    
    test_collection = select_collection('mnist_db', 'mnist_test')
    x_test, y_test = load_image_data(test_collection)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    return x_train, x_test, y_train, y_test


def create_model(layer_depth, layer_size, activation_function):
    """
    Creates a neural network model with the specified hyperparameters.

    Args:
        layer_depth (int): Number of hidden layers.
        layer_size (int): Number of units per hidden layer.
        activation_function (str): Activation function to use in the hidden layers.

    Returns:
        keras.Sequential: A compiled Keras model.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28 * 28,)))

    for _ in range(layer_depth):
        model.add(
            keras.layers.Dense(layer_size, activation=activation_function)
            )

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model


def train_and_evaluate_model(
    hyperparameters,
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    Trains and evaluates a model using the given hyperparameters.

    Args:
        hyperparameters (dict): Dictionary containing model hyperparameters.
        x_train (numpy.ndarray): Training data images.
        y_train (numpy.ndarray): Training data labels.
        x_test (numpy.ndarray): Validation data images.
        y_test (numpy.ndarray): Validation data labels.

    Returns:
        tuple: A tuple containing:
            model (keras.Sequential): Trained Keras model.
            test_acc (float): Validation accuracy of the model.
    """
    model = create_model(
        layer_depth=hyperparameters['Depth'],
        layer_size=hyperparameters['Size'],
        activation_function=hyperparameters['Activation'],
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    return model, test_acc



def update_for_improvement(model, accuracy_score):
    """
    Updates the model in the database if the new model shows improvement.

    Args:
        model (keras.Sequential): The newly trained model.
        accuracy_score (float): Accuracy of the new model.

    Returns:
        keras.Sequential: The selected model (either the new or the previous best).
    """
    
    collection = select_collection(
        'model_db',
        'model_collection'
    )

    last_model = collection.find_one(sort=[('_id', -1)])

    if last_model is None:
        print(f"No model pickle found in the collection.")
        return model
    if last_model['accuracy_score'] > accuracy_score:
        print('worsening')
    else:
        pickle_path = pickle_model(model)
        upload_model_to_mongo(
            pickle_path,
            accuracy_score
        )
    
    # old_model = pickle.loads(newest_document['model_data'])


def pickle_model(model):
    """
    Saves the given model as a pickle file.

    Args:
        model (keras.Sequential): The Keras model to save.

    Returns:
        str: The filename of the saved pickle file.
    """
    model_filename = 'trained_model.pickle'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    return model_filename

def upload_model_to_mongo(model_filename, accuracy_score):
    """
    Uploads the pickled model to MongoDB along with its metadata.

    Args:
        model_filename (str): The filename of the pickled model.
        accuracy_score (float): The accuracy score of the model.
    """
    collection = select_collection('model_db', 'model_collection')

    with open(model_filename, 'rb') as f:
        model_data = f.read()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_document = {
        'model_name': model_filename,
        'model_data': model_data,
        'timestamp': timestamp,
        'accuracy_score': accuracy_score
    }
    
    collection.insert_one(model_document)
    print(f'Model {model_filename} uploaded to MongoDB successfully.')

def final_prediction():
    """
    Loads hyperparameters from MongoDB, trains a model with the best hyperparameters, and updates the model if improved.

    The model is trained using either the hyperparameters for the ReLU or Sigmoid activation function, based on validation accuracy.
    
    Returns:
        None
    """
    relu_hyperparameters =  load_hyperparameters('relu')
    sigmoid_hyperparameters =  load_hyperparameters('sigmoid')
    x_train, x_test, y_train, y_test = load_data()
    sigmoid_better = (
        sigmoid_hyperparameters['Val Accuracy']
        > relu_hyperparameters['Val Accuracy']
    )
    if sigmoid_better:
        hyperparameter_dict = sigmoid_hyperparameters
    else:
        hyperparameter_dict = relu_hyperparameters
    
    model, accuracy_score = train_and_evaluate_model(
        hyperparameter_dict,
        x_train,
        y_train,
        x_test,
        y_test
    )
    
    model = update_for_improvement(model, accuracy_score)
    
    
    
    
if __name__ == "__main__":
    final_prediction()
        
    