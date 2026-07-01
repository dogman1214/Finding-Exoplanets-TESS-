import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_1d_cnn_model(input_length=2000):
    model = models.Sequential([
        layers.Conv1D(16, 7, activation='relu', input_shape=(input_length, 1),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3),
        layers.Dropout(0.5),
        
        layers.Conv1D(32, 5, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3),
        layers.Dropout(0.5),
        
        layers.Conv1D(64, 3, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def generate_synthetic_synthetic_transits(n_samples=1000, length=2000):
    logger.info(f"Generating {n_samples} synthetic transit samples...")
    
    X = np.random.normal(0, 0.1, (n_samples, length, 1))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        transit_depth = np.random.uniform(0.005, 0.05)
        transit_width = np.random.randint(50, 150)
        transit_center = np.random.randint(200, length-200)
        
        signal = np.ones(length)
        
        ingress_width = transit_width // 3
        for j in range(ingress_width):
            x = j / ingress_width
            signal[transit_center - transit_width//2 + j] = 1 - transit_depth * x
        
        flat_width = transit_width - 2 * ingress_width
        for j in range(flat_width):
            signal[transit_center - transit_width//2 + ingress_width + j] = 1 - transit_depth
        
        for j in range(ingress_width):
            x = j / ingress_width
            signal[transit_center + transit_width//2 - ingress_width + j] = 1 - transit_depth * (1 - x)
        
        X[i, :, 0] += signal
        y[i] = 1
    
    return X, y

def load_real_dataset(pos_path='confirmed_positive_vectors.npy', neg_path='negative_vectors.npy'):
    logger.info(f"Loading real dataset from {pos_path} and {neg_path}")
    pos = np.load(pos_path)
    neg = np.load(neg_path)
    
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    
    if len(X.shape) == 2:
        X = X[..., np.newaxis]
        
    return X, y

if __name__ == "__main__":
    import os
    
    pos_path = 'confirmed_positive_vectors.npy'
    neg_path = 'negative_vectors.npy'
    
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        logger.info(f"Loading real training data from '{pos_path}' and '{neg_path}'...")
        X, y = load_real_dataset(pos_path, neg_path)
        model_filename = 'exoplanet_cnn_real.h5'
        history_plot_filename = 'training_history_real.png'
    else:
        logger.info("Real dataset files not found.")
        sys.exit(0)
        
    logger.info(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    logger.info(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    counts = np.bincount(y_train.astype(int))
    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    class_weight = {0: weight_for_0 * (len(y_train)/2), 1: weight_for_1 * (len(y_train)/2)}
    logger.info(f"Calculated class weights: {class_weight}")

    model = create_1d_cnn_model(input_length=X.shape[1])
    logger.info("\nModel architecture summary generated.")
    model.summary()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filename, monitor='val_precision', save_best_only=True, mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )
    
    model.save(model_filename)
    logger.info(f"Model saved as '{model_filename}'")
    
    logger.info("Generating training history plots...")
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    prec_key = [k for k in history.history.keys() if 'precision' in k]
    plt.subplot(1, 4, 3)
    if prec_key:
        plt.plot(history.history[prec_key[0]], label='Train Prec')
        plt.plot(history.history['val_' + prec_key[0]], label='Val Prec')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    rec_key = [k for k in history.history.keys() if 'recall' in k]
    plt.subplot(1, 4, 4)
    if rec_key:
        plt.plot(history.history[rec_key[0]], label='Train Rec')
        plt.plot(history.history['val_' + rec_key[0]], label='Val Rec')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(history_plot_filename)
    logger.info(f"Training history saved as '{history_plot_filename}'")

    