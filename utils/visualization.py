import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import Config

def plot_training_progress(g_losses, d_losses, save_path="training_progress.png"):
    """Plot generator and discriminator loss during training"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title("Training Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def generate_sample_images(generator, mapper, test_images, test_ages, config, epoch, n_samples=4):
    """Generate sample aged images for visualization"""
    z = tf.random.normal([n_samples, config.STYLE_DIM])
    styles = mapper([z, test_ages[:n_samples]])
    
    generated = generator([test_images[:n_samples], styles])
    
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i+1)
        plt.imshow((test_images[i] * 127.5 + 127.5).numpy().astype(np.uint8))
        plt.title(f"Original\nAge: {test_ages[i]}")
        plt.axis('off')
        
        # Aged
        plt.subplot(2, n_samples, i+n_samples+1)
        plt.imshow((generated[i] * 127.5 + 127.5).numpy().astype(np.uint8))
        plt.title(f"Aged to {test_ages[i]+20}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.SAMPLE_DIR}/epoch_{epoch:04d}.png")
    plt.close()

def create_aging_progression(generator, mapper, test_image, config):
    """Create a progression of aging from young to old"""
    ages = np.linspace(config.AGE_RANGE[0], config.AGE_RANGE[1], 8)
    z = tf.random.normal([1, config.STYLE_DIM])
    
    plt.figure(figsize=(20, 5))
    for i, age in enumerate(ages):
        style = mapper([z, [age]])
        aged_img = generator([test_image[tf.newaxis, ...], style])[0]
        
        plt.subplot(1, len(ages), i+1)
        plt.imshow((aged_img * 127.5 + 127.5).numpy().astype(np.uint8))
        plt.title(f"Age {int(age)}")
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()