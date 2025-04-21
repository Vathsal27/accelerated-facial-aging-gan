import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

class StarDiscriminator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Kernel initializer
        kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # Shared layers with spectral normalization
        self.shared = tf.keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same',
                         kernel_initializer=kernel_init,
                         use_bias=False),
            layers.LeakyReLU(0.2),
            
            SpectralNormalization(
                layers.Conv2D(128, 4, strides=2, padding='same',
                             kernel_initializer=kernel_init,
                             use_bias=False)
            ),
            layers.LeakyReLU(0.2),
            
            SpectralNormalization(
                layers.Conv2D(256, 4, strides=2, padding='same',
                             kernel_initializer=kernel_init,
                             use_bias=False)
            ),
            layers.LeakyReLU(0.2),
            
            SpectralNormalization(
                layers.Conv2D(512, 4, strides=2, padding='same',
                             kernel_initializer=kernel_init,
                             use_bias=False)
            ),
            layers.LeakyReLU(0.2)
        ])
        
        # Output layers
        self.src = SpectralNormalization(
            layers.Conv2D(1, 3, padding='same',
                         kernel_initializer=kernel_init)
        )
        
        self.cls = SpectralNormalization(
            layers.Dense(config.NUM_DOMAINS,
                        kernel_initializer=kernel_init)
        )
        
        # Minibatch discrimination (optional)
        self.minibatch_layer = MinibatchDiscrimination(
            num_kernels=100, 
            dim_per_kernel=5
        )

    def call(self, x, training=None):
        features = self.shared(x)
        
        # Add minibatch discrimination features
        mb_features = self.minibatch_layer(features)
        features = tf.concat([features, mb_features], axis=-1)
        
        validity = self.src(features)
        label = self.cls(layers.Flatten()(features))
        
        return validity, label


# Helper Classes ==============================================

class SpectralNormalization(layers.Wrapper):
    """Spectral normalization layer wrapper for GAN stability"""
    def __init__(self, layer, iteration=1, eps=1e-12, **kwargs):
        super().__init__(layer, **kwargs)
        self.iteration = iteration
        self.eps = eps
        
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer='random_normal',
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )
        
    def call(self, inputs):
        self._compute_weights()
        output = self.layer(inputs)
        return output
        
    def _compute_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u
        
        for _ in range(self.iteration):
            v = tf.math.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
            
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        w_bar = self.w / sigma
        
        self.layer.kernel.assign(w_bar)
        self.u.assign(u)


class MinibatchDiscrimination(layers.Layer):
    """Minibatch discrimination layer for mode collapse prevention"""
    def __init__(self, num_kernels=100, dim_per_kernel=5):
        super().__init__()
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel
        
    def build(self, input_shape):
        self.T = self.add_weight(
            shape=[input_shape[-1], self.num_kernels * self.dim_per_kernel],
            initializer='random_normal',
            trainable=True,
            name='minibatch_T'
        )
        
    def call(self, inputs):
        # Compute minibatch features
        x = tf.matmul(inputs, self.T)
        x = tf.reshape(x, [-1, self.num_kernels, self.dim_per_kernel])
        
        # Compute L1 distances between samples
        diffs = tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        
        # Sum over features
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        
        return minibatch_features