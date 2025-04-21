import gradio as gr
import tensorflow as tf
import numpy as np
from config import Config
from models.generator import StarGenerator
from models.mapper import StyleMapper
from utils.evaluation import AgeEvaluator

class AgingApp:
    def __init__(self, config):
        self.config = config
        self.generator = StarGenerator(config)
        self.mapper = StyleMapper(config)
        self.age_evaluator = AgeEvaluator(config)
        
        # Load weights
        self.load_weights()
        
    def load_weights(self):
        latest = tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
        if latest:
            checkpoint = tf.train.Checkpoint(
                generator=self.generator,
                mapper=self.mapper
            )
            checkpoint.restore(latest)
            print(f"Loaded weights from {latest}")
            
    def preprocess_image(self, image):
        image = tf.image.resize(image, [self.config.IMG_SIZE, self.config.IMG_SIZE])
        image = (image - 127.5) / 127.5  # Normalize
        return image
        
    def predict_aging(self, input_image, target_age, wrinkles, gray_hair):
        # Preprocess
        input_image = self.preprocess_image(input_image)
        
        # Estimate original age
        original_age = self.age_evaluator.model.predict(input_image[tf.newaxis, ...])[0][0]
        
        # Generate style vector with aging controls
        z = tf.random.normal([1, self.config.STYLE_DIM])
        style = self.mapper([z, [target_age]])
        
        # Modify style based on controls
        style = self.adjust_aging_effects(style, wrinkles, gray_hair)
        
        # Generate aged image
        aged_image = self.generator([input_image[tf.newaxis, ...], style])[0]
        aged_image = (aged_image * 127.5 + 127.5).numpy().astype(np.uint8)
        
        # Evaluate
        evaluation = self.age_evaluator.evaluate(
            input_image, 
            (aged_image - 127.5) / 127.5, 
            original_age
        )
        
        return aged_image, evaluation
        
    def adjust_aging_effects(self, style, wrinkles, gray_hair):
        # Simple linear adjustment of style vectors
        style = style.numpy()
        style[:, :32] *= (1 + wrinkles * 0.5)  # Wrinkles affect first half
        style[:, 32:] *= (1 + gray_hair * 0.3)  # Gray hair affects second half
        return tf.convert_to_tensor(style)

def create_interface():
    config = Config()
    app = AgingApp(config)
    
    interface = gr.Interface(
        fn=app.predict_aging,
        inputs=[
            gr.Image(label="Input Face", shape=(256, 256)),
            gr.Slider(10, 80, value=30, label="Target Age"),
            gr.Slider(0, 1, value=0.5, step=0.1, label="Wrinkles Intensity"),
            gr.Slider(0, 1, value=0.5, step=0.1, label="Gray Hair Amount")
        ],
        outputs=[
            gr.Image(label="Aged Face"),
            gr.JSON(label="Aging Evaluation")
        ],
        title="Face Aging with StarGAN v2",
        description="Predict how a face will age with controllable parameters"
    )
    return interface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()