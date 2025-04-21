class Config:
    # Dataset
    DATA_PATH = "data/processed"
    IMG_SIZE = 256
    BATCH_SIZE = 8
    AGE_RANGE = (10, 80)  # Min and max age to consider
    
    # Model architecture
    STYLE_DIM = 64
    MAX_CONV_DIM = 512
    NUM_DOMAINS = 6  # Age groups
    
    # Training
    EPOCHS = 1000
    LR = 1e-4
    BETA1 = 0.0
    BETA2 = 0.99
    LAMBDA_STYLE = 1.0
    LAMBDA_CYC = 1.0
    LAMBDA_DS = 1.0
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    SAMPLE_DIR = "samples"
    LOG_DIR = "logs"
    
    # Evaluation
    AGE_MODEL_PATH = "models/age_classifier.h5"