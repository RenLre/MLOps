import numpy as np
import tensorflow as tf
from mongodb_function import select_collection, overwrite_image_data


def manipulate_fold(collection, images, labels, phase):
    """
    Manipulates the dataset based on the specified phase and updates it in MongoDB.

    Args:
        collection (pymongo.collection.Collection): MongoDB collection where data will be overwritten.
        images (numpy.ndarray): The images to be manipulated.
        labels (numpy.ndarray): The labels corresponding to the images.
        phase (str): Specifies the manipulation to be applied. 
                     'half' reduces the dataset by half,
                     'full' uses the full dataset,
                     'noisy' adds Gaussian noise to the images.
    """
    if phase == 'half':
        half = lambda x, y: (x[:len(x)//2], y[:len(y)//2])
        halved_images, halved_labels = half(images, labels)
        overwrite_image_data(
            collection,
            halved_images,
            halved_labels
        )
    
    elif phase == 'full':
        overwrite_image_data(
            collection,
            images,
            labels
        )
        
    elif phase == 'noisy':
        noise = np.random.normal(0, 0.1, images.shape) 
        noisy_images = images + noise
        noisy_images = np.clip(noisy_images, 0, 1)
        overwrite_image_data(
            collection,
            noisy_images,
            labels
        )
    

def manipulate_data(phase='full'):
    """
    Loads the MNIST dataset and applies the specified manipulation phase
    to both the training and test data.
    
    Args:
        phase (str, optional): Specifies the manipulation to apply to the dataset. Defaults to 'full'.
                               Options:
                               'half' to halve the dataset,
                               'full' to keep the dataset intact,
                               'noisy' to add noise to the images.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    collection_train = select_collection('mnist_db', 'mnist_train')
    manipulate_fold(collection_train, x_train, y_train, phase)
    
    collection_test = select_collection('mnist_db', 'mnist_test')
    manipulate_fold(collection_test, x_test, y_test, phase)
    
        
if __name__ == "__main__":
    manipulate_data('half')
    