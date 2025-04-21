import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

class AgeEvaluator:
    def __init__(self, config):
        self.model = self.build_age_classifier(config)
        
    def build_age_classifier(self, config):
        base = EfficientNetB0(include_top=False, weights='imagenet', 
                             input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        base.trainable = False
        
        inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        x = base(inputs)
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def evaluate(self, original_img, aged_img, original_age):
        aged_age = self.model.predict(aged_img[tf.newaxis, ...])[0][0]
        return {
            'predicted_age': float(aged_age),
            'age_difference': float(aged_age - original_age),
            'realism_score': self.calculate_realism(aged_img)
        }
        
    def calculate_realism(self, image):
        # Placeholder for a realism scoring metric
        return 0.0