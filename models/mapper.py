import tensorflow as tf
from tensorflow.keras import layers

class StyleMapper(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        dim = 512
        for _ in range(4):
            layers.extend([
                layers.Dense(dim),
                layers.LeakyReLU(0.2)
            ])
            dim //= 2
            
        self.net = tf.keras.Sequential(layers)
        self.final = layers.Dense(config.STYLE_DIM * 2)  # Mean and variance
        
    def call(self, inputs):
        z, age = inputs
        x = tf.concat([z, tf.cast(age, tf.float32)], axis=-1)
        return self.final(x)