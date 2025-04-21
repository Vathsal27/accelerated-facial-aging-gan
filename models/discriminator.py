import tensorflow as tf
from tensorflow.keras import layers

class StarDiscriminator(tf.keras.Model):
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
        
        # Output layers
        self.src = layers.Conv2D(1, 3, padding='same')
        self.cls = layers.Dense(config.NUM_DOMAINS)
        
    def call(self, x, training=None):
        features = self.shared(x)
        validity = self.src(features)
        label = self.cls(layers.Flatten()(features))
        return validity, label