import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

class AgeEstimator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base network
        self.base = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
        )
        self.base.trainable = False
        
        # Regression head
        self.head = tf.keras.Sequential([
            layers.GlobalAvgPool2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
    def call(self, inputs, training=None):
        features = self.base(inputs)
        age = self.head(features)
        return age
        
    def compile(self, **kwargs):
        super().compile(
            loss='mse',
            metrics=['mae'],
            **kwargs
        )
        
    def estimate_age(self, image):
        """Convenience method for single image age estimation"""
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        return self(image).numpy()[0][0]