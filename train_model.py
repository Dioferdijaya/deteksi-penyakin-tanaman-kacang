"""
Script Training Model - Deteksi Penyakit Daun Kacang
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Activation, GlobalAveragePooling2D
)

# Paths
DATA_DIR = '.'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
MODEL_PATH = 'model_penyakit_daun_kacang_fixed.h5'
IMG_SIZE = 150

def load_data(csv_file, data_dir=DATA_DIR):
    """Load data dari CSV"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row['image:FILE'])
        raw_label = int(row['category'])
        label = 0 if raw_label == 0 else 1
        
        if os.path.exists(img_path):
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        else:
            print(f"File not found: {img_path}")
    
    return np.array(images), np.array(labels)

def build_model():
    """Build CNN model"""
    print("\nBuilding model...")
    
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')  # Binary: Healthy=0, Diseased=1
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model():
    """Main training function"""
    print("="*60)
    print("Training CNN Model - Deteksi Penyakit Daun Kacang")
    print("="*60)
    
    # Load training data
    print("\n[1/5] Loading training data...")
    X_train, y_train = load_data(TRAIN_CSV)
    print(f"✓ Loaded {len(X_train)} training images")
    print(f"  Shape: {X_train.shape}, Labels: {np.unique(y_train, return_counts=True)}")
    
    # Load validation data
    print("\n[2/5] Loading validation data...")
    X_val, y_val = load_data(VAL_CSV)
    print(f"✓ Loaded {len(X_val)} validation images")
    print(f"  Shape: {X_val.shape}, Labels: {np.unique(y_val, return_counts=True)}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ No data found! Check file paths.")
        return False
    
    # Build model
    print("\n[3/5] Building model...")
    model = build_model()
    
    # Train
    print("\n[4/5] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        verbose=1
    )
    
    # Save
    print("\n[5/5] Saving model...")
    model.save(MODEL_PATH, save_format='h5')
    print(f"✓ Model saved to {MODEL_PATH}")
    
    # Evaluate
    print("\nModel Evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    return True

if __name__ == '__main__':
    train_model()
