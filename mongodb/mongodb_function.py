import tensorflow as tf
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np



ef connect_db():
    """
    Establishes a connection to a MongoDB database.

    Returns:
        MongoClient: A MongoDB client object for database interactions.
    """
    uri = "mongodb+srv://gerkeleon:P6QEXVEa90gVRmZQ@clusterml.l2kyy.mongodb.net/?retryWrites=true&w=majority&appName=ClusterML"


    client = MongoClient(uri, server_api=ServerApi('1'), socketTimeoutMS=60000, connectTimeoutMS=60000)

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        
    return client


def select_collection(database_name, collection_name):
    """
    Selects a MongoDB collection from a specified database.

    Args:
        database_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection within the database.

    Returns:
        Collection: The selected MongoDB collection.
    """
    client = connect_db()

    db = client[database_name]
    collection = db[collection_name]
    
    return collection


def insert_image_data(collection, images, labels):
    """
    Inserts image data and labels into a MongoDB collection.

    Args:
        collection (Collection): The MongoDB collection to insert data into.
        images (numpy.ndarray): The image data to insert.
        labels (numpy.ndarray): The corresponding labels for the images.

    Returns:
        None
    """
    data_to_insert = []
    for i in range(len(images)):
        document = {
            'image': images[i].tolist(),
            'label': int(labels[i]),
        }
        data_to_insert.append(document)
    
    result = collection.insert_many(data_to_insert)
    print(f'{len(result.inserted_ids)} Dokumente wurden in die MongoDB eingefügt.')
    
    
def overwrite_image_data(
    collection,
    images,
    labels
):
    """
    Overwrites all image data in a MongoDB collection with new images and labels.

    Args:
        collection (Collection): The MongoDB collection to overwrite.
        images (numpy.ndarray): The new image data.
        labels (numpy.ndarray): The new corresponding labels.

    Returns:
        None
    """
    collection.delete_many({})
    insert_image_data(collection, images, labels)
    
    
def load_image_data(collection):
    """
    Loads image data and labels from a MongoDB collection.

    Args:
        collection (Collection): The MongoDB collection to load data from.

    Returns:
        tuple: A tuple containing:
            - images (numpy.ndarray): The loaded image data.
            - labels (numpy.ndarray): The corresponding labels for the images.
    """
    cursor = collection.find()
    images = []
    labels = []
    for document in cursor:
        images.append(np.array(document['image']))
        labels.append(document['label'])

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


def insert_hyperparameter_json(best_parameter_dict, activation):
    """
    Inserts a JSON document of hyperparameters into a MongoDB collection based on activation function.

    Args:
        best_parameter_dict (dict): The hyperparameter configuration to insert.
        activation (str): The activation function used.

    Returns:
        None
    """
    collection_name = 'hyperparam_' + activation
    collection = select_collection(
        'hyperparam_db',
        collection_name
    )
    
    result = collection.insert_one(
        best_parameter_dict
    )
    
    print(f'JSON für {activation} wurde in die MongoDB eingefügt.')
    

def load_hyperparameters(activation):
    """
    Loads the most recent hyperparameters from a MongoDB collection based on the activation function.

    Args:
        activation (str): The activation function (e.g., 'relu', 'sigmoid').

    Returns:
        dict: The most recent hyperparameter configuration from the MongoDB collection.
    """
    collection_name = 'hyperparam_' + activation
    collection = select_collection(
        'hyperparam_db',
        collection_name
    )

    newest_document = collection.find_one(sort=[('_id', -1)])  # Sortiere nach _id absteigend

    return newest_document