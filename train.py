import os
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from config import Config
from models.generator import StarGenerator
from models.discriminator import StarDiscriminator
from models.mapper import StyleMapper
from utils.data_loader import UTKFaceDataset
from utils.evaluation import AgeEvaluator

class StarGANTrainer:
    def __init__(self, config):
        self.config = config
        self.generator = StarGenerator(config)
        self.mapper = StyleMapper(config)
        self.discriminator = StarDiscriminator(config)
        self.age_evaluator = AgeEvaluator(config)
        
        # Optimizers
        self.g_optimizer = AdamW(config.LR, beta_1=config.BETA1, beta_2=config.BETA2)
        self.d_optimizer = AdamW(config.LR, beta_1=config.BETA1, beta_2=config.BETA2)
        
        # Checkpoints
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            mapper=self.mapper,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, config.CHECKPOINT_DIR, max_to_keep=3
        )
        
    def compute_g_loss(self, real_images, target_labels):
        # Style generation
        z = tf.random.normal([real_images.shape[0], self.config.STYLE_DIM])
        style = self.mapper([z, target_labels])
        
        # Generate fake images
        fake_images = self.generator([real_images, style])
        
        # Discriminator outputs
        d_fake, cls_fake = self.discriminator(fake_images)
        
        # Adversarial loss
        g_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_fake))
        
        # Classification loss
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            target_labels, cls_fake, from_logits=True
        )
        cls_loss = tf.reduce_mean(cls_loss)
        
        # Style reconstruction loss
        style_recon_loss = tf.reduce_mean(tf.abs(style - self.mapper([z, target_labels])))
        
        # Total loss
        total_loss = g_loss + cls_loss + style_recon_loss
        return total_loss
        
    def train_step(self, batch):
        real_images, labels = batch
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generator loss
            g_loss = self.compute_g_loss(real_images, labels)
            
            # Discriminator loss
            z = tf.random.normal([real_images.shape[0], self.config.STYLE_DIM])
            style = self.mapper([z, labels])
            fake_images = self.generator([real_images, style])
            
            d_real, cls_real = self.discriminator(real_images)
            d_fake, _ = self.discriminator(fake_images)
            
            d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
            d_loss = d_loss_real + d_loss_fake
            
        # Apply gradients
        g_vars = self.generator.trainable_variables + self.mapper.trainable_variables
        d_vars = self.discriminator.trainable_variables
        
        g_grads = g_tape.gradient(g_loss, g_vars)
        d_grads = d_tape.gradient(d_loss, d_vars)
        
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))
        self.d_optimizer.apply_gradients(zip(d_grads, d_vars))
        
        return {'g_loss': g_loss, 'd_loss': d_loss}
        
    def train(self):
        # Load dataset
        dataset = UTKFaceDataset(self.config).create_dataset()
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            
            for step, batch in enumerate(dataset):
                losses = self.train_step(batch)
                
                if step % 100 == 0:
                    print(f"Step {step}: G Loss: {losses['g_loss']:.4f}, D Loss: {losses['d_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.manager.save()
                print(f"Saved checkpoint at epoch {epoch + 1}")
                
            # TODO: Add sample generation and evaluation

if __name__ == "__main__":
    config = Config()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    trainer = StarGANTrainer(config)
    trainer.train()