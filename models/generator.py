import tensorflow as tf
from tensorflow.keras import layers, initializers

class AdaptiveInstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        
    def build(self, input_shape):
        # Add learnable scale and bias parameters
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            name='adain_gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            name='adain_beta'
        )
        
    def call(self, x, style):
        # Calculate mean and variance
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (x - mean) / std
        
        # Split style into mean and std components
        style_mean, style_std = tf.split(style, 2, axis=-1)
        
        # Apply style with learned parameters
        out = (normalized * (1 + self.gamma) * style_std) + (style_mean + self.beta)
        return out

class ResBlock(layers.Layer):
    def __init__(self, filters, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.norm1 = AdaptiveInstanceNorm()
        self.conv1 = layers.Conv2D(
            filters, 3, padding='same',
            kernel_initializer=initializers.RandomNormal(0, 0.02),
            use_bias=False
        )
        self.norm2 = AdaptiveInstanceNorm()
        self.conv2 = layers.Conv2D(
            filters, 3, padding='same',
            kernel_initializer=initializers.RandomNormal(0, 0.02),
            use_bias=False
        )
        self.skip = layers.Conv2D(
            filters, 1, padding='same',
            kernel_initializer=initializers.RandomNormal(0, 0.02),
            use_bias=False
        )
        
    def call(self, inputs):
        x, style = inputs
        
        # First normalization and activation
        x = self.norm1(x, style)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        # Upsample if needed
        if self.upsample:
            x = tf.image.resize(x, [x.shape[1]*2, x.shape[2]*2], method='nearest')
        
        # First convolution
        x = self.conv1(x)
        
        # Second normalization and activation
        x = self.norm2(x, style)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        # Second convolution
        x = self.conv2(x)
        
        # Skip connection
        if self.upsample:
            skip = tf.image.resize(inputs[0], [x.shape[1], x.shape[2]], method='nearest')
        else:
            skip = inputs[0]
        skip = self.skip(skip)
        
        return x + skip * 0.1  # Weighted residual connection

class StarGenerator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial layers with reflection padding
        self.initial_conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')),
            layers.Conv2D(
                64, 7, padding='valid',
                kernel_initializer=initializers.RandomNormal(0, 0.02),
                use_bias=False
            ),
            AdaptiveInstanceNorm(),
            layers.LeakyReLU(alpha=0.2)
        ])
        
        # Downsample blocks
        self.down_blocks = [
            ResBlock(128),
            ResBlock(256),
            ResBlock(512)
        ]
        
        # Bottleneck blocks
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
        
        # Final output layers
        self.final_conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')),
            layers.Conv2D(
                3, 7, padding='valid', activation='tanh',
                kernel_initializer=initializers.RandomNormal(0, 0.02)
            )
        ])
        
        # Style processing
        self.style_dense = layers.Dense(512 * 2)  # For style modulation
        
    def call(self, inputs, training=None):
        x, style = inputs
        
        # Process style vector
        style = self.style_dense(style)
        styles = tf.split(
            style, 
            num_or_size_splits=len(self.down_blocks) + len(self.bottleneck.layers) + len(self.up_blocks),
            axis=-1
        )
        style_idx = 0
        
        # Initial processing
        x = self.initial_conv(x)
        
        # Downsample path
        for block in self.down_blocks:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Bottleneck path
        for block in self.bottleneck.layers:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Upsample path
        for block in self.up_blocks:
            x = block([x, styles[style_idx]])
            style_idx += 1
            
        # Final output
        return self.final_conv(x)