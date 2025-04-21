import tensorflow as tf
from tensorflow.keras import layers
from config import Config

class StyleEncoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared layers
        self.shared = tf.keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(256, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2)
        ])
        
        # Style extraction
        self.style_extractor = tf.keras.Sequential([
            layers.GlobalAvgPool2D(),
            layers.Dense(config.STYLE_DIM * 2)  # For mean and variance
        ])
        
        # Age classifier
        self.age_classifier = layers.Dense(config.NUM_DOMAINS)
        
    def call(self, inputs, training=None):
        features = self.shared(inputs)
        
        # Style code (mean and log variance)
        style = self.style_extractor(features)
        mean, logvar = tf.split(style, 2, axis=-1)
        
        # Age prediction
        age_logits = self.age_classifier(features)
        
        return mean, logvar, age_logits
        
    def encode(self, images):
        """Encode images to style vectors"""
        mean, logvar, _ = self(images)
        # Reparameterization trick
        epsilon = tf.random.normal(tf.shape(mean))
        style = mean + tf.exp(0.5 * logvar) * epsilon
        return style