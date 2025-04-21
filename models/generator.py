import tensorflow as tf
from tensorflow.keras import layers

class AdaptiveInstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        
    def call(self, x, style):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (x - mean) / std
        
        style_mean, style_std = tf.split(style, 2, axis=-1)
        out = normalized * style_std + style_mean
        return out

class ResBlock(layers.Layer):
    def __init__(self, filters, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.norm1 = AdaptiveInstanceNorm()
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.norm2 = AdaptiveInstanceNorm()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.skip = layers.Conv2D(filters, 1, padding='same')
        
    def call(self, inputs):
        x, style = inputs
        x = self.norm1(x, style)
        x = tf.nn.relu(x)
        if self.upsample:
            x = tf.image.resize(x, [x.shape[1]*2, x.shape[2]*2])
        x = self.conv1(x)
        
        x = self.norm2(x, style)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if self.upsample:
            skip = tf.image.resize(inputs[0], [x.shape[1], x.shape[2]])
            skip = self.skip(skip)
        else:
            skip = self.skip(inputs[0])
            
        return x + skip

class StarGenerator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial layers
        self.downsample = tf.keras.Sequential([
            layers.Conv2D(64, 7, padding='same'),
            AdaptiveInstanceNorm(),
            layers.ReLU()
        ])
        
        # Downsample blocks
        self.down_blocks = [
            ResBlock(128),
            ResBlock(256),
            ResBlock(512)
        ]
        
        # Bottleneck
        self.bottleneck = tf.keras.Sequential([
            ResBlock(512),
            ResBlock(512),
            ResBlock(512)
        ])
        
        # Upsample blocks
        self.up_blocks = [
            ResBlock(256, upsample=True),
            ResBlock(128, upsample=True),
            ResBlock(64, upsample=True)
        ]
        
        # Final layers
        self.final_conv = layers.Conv2D(3, 7, padding='same', activation='tanh')
        
    def call(self, inputs, training=None):
        x, style = inputs
        styles = tf.split(style, len(self.down_blocks) + len(self.bottleneck.layers) + len(self.up_blocks), axis=1)
        style_idx = 0
        
        # Downsample
        x = self.downsample(x)
        for block in self.down_blocks:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Bottleneck
        for block in self.bottleneck.layers:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Upsample
        for block in self.up_blocks:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Final output
        x = self.final_conv(x)
        return x