import os
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

class UTKFaceDataset:
    # Modify the UTKFaceDataset class __init__ method:
    def __init__(self, config):
        self.config = config
        self.image_paths = []
        self.ages = []
        
        for img_name in os.listdir(config.DATA_PATH):
            if img_name.endswith(".chip.jpg"):
                try:
                    # Extract age from filename like "54_0_3_20170119184252470.jpg.chip.jpg"
                    age = int(img_name.split('_')[0])
                    if config.AGE_RANGE[0] <= age <= config.AGE_RANGE[1]:
                        self.image_paths.append(os.path.join(config.DATA_PATH, img_name))
                        self.ages.append(age)
                except (ValueError, IndexError):
                    continue
                
        # Create age groups
        self.age_bins = np.linspace(
            config.AGE_RANGE[0], 
            config.AGE_RANGE[1], 
            config.NUM_DOMAINS + 1
        )
        self.age_labels = np.digitize(self.ages, self.age_bins[:-1]) - 1
        
    def preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.config.IMG_SIZE, self.config.IMG_SIZE])
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        return img
        
    def create_dataset(self):
        def _parse_image(idx):
            image = self.preprocess_image(self.image_paths[idx])
            label = self.age_labels[idx]
            return image, label
            
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(self.image_paths)))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset