import tensorflow as tf
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np



def connect_db():
    # MongoDB-Verbindung herstellen
    uri = "mongodb+srv://gerkeleon:P6QEXVEa90gVRmZQ@clusterml.l2kyy.mongodb.net/?retryWrites=true&w=majority&appName=ClusterML"


    client = MongoClient(uri, server_api=ServerApi('1'), socketTimeoutMS=60000, connectTimeoutMS=60000)

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        
    return client


def select_collection(database_name, collection_name):
    client = connect_db()

    # Auswahl der Datenbank und der Sammlung
    db = client[database_name]  # Name der Datenbank
    collection = db[collection_name]  # Name der Sammlung
    
    return collection

def insert_image_data(collection, images, labels, database_name, collection_name):
    collection = select_collection(database_name, collection_name)
    data_to_insert = []
    for i in range(len(images)):
        document = {
            'image': images[i].tolist(),  # Bild als Liste (da MongoDB keine NumPy-Arrays speichert)
            'label': int(labels[i]),      # Label als Integer
        }
        data_to_insert.append(document)
    
    result = collection.insert_many(data_to_insert)
    print(f'{len(result.inserted_ids)} Dokumente wurden in die MongoDB eingefügt.')
    
    
def overwrite_image_data(
    images,
    labels,
    dataset_name,
    database_name,
    collection_name
):
    ...
    
def load_image_data(database_name, collection_name):
    collection = select_collection(database_name, collection_name)
    # def load_mnist_data(option='full', limit=10000):
    cursor = collection.find()  # Eine Teilmenge der Daten laden
    images = []
    labels = []
    for document in cursor:
        images.append(np.array(document['image']))  # Bild als Array
        labels.append(document['label'])  # Label

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def insert_hyperparameter_json(best_parameter_dict):
    collection = select_collection(
        'hyperparam_db',
        'hyperparam_collection'
    )
    
    result = collection.insert_one(
        best_parameter_dict
    )
    
    print(f'{len(result.inserted_id)} Dokumente wurden in die MongoDB eingefügt.')
    

def load_hyperparameters(most_recent=True):
    # Erstelle die Datenbank "castro" und die Collection "mnist"
    
    collection = select_collection(
        'hyperparam_db',
        'hyperparam_collection'
    )

    # Finde den neuesten Datenpunkt (sortiert nach _id, der automatisch eingefügte ObjectId)
    newest_document = collection.find_one(sort=[('_id', -1)])  # Sortiere nach _id absteigend

    # Gebe den neuesten Datenpunkt aus
    return newest_document
#col