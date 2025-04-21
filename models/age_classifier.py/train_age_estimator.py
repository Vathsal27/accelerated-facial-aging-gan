import tensorflow as tf
from .age_estimator import AgeEstimator
from config import Config
from utils.data_loader import UTKFaceDataset

def train_age_classifier():
    config = Config()
    model = AgeEstimator(config)
    
    # Load dataset
    dataset = UTKFaceDataset(config).create_dataset()
    
    # Split dataset (modify as needed)
    train_size = int(0.8 * len(dataset))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    # Compile and train
    model.compile(optimizer='adam')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    # Save model
    model.save_weights(f"{config.AGE_MODEL_PATH}")
    
if __name__ == "__main__":
    train_age_classifier()