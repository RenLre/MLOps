import time

def train_model(hyperparameters):
    print("Starting Model Training process...")
    print(f"Using hyperparameters: {hyperparameters}")
    time.sleep(10)  # Simulating model training
    print("Model Training completed.")

if __name__ == "__main__":
    # This is just for testing the script independently
    test_hyperparameters = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100
    }
    train_model(test_hyperparameters)