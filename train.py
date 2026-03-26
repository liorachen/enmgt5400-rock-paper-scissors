# train.py — Rock Paper Scissors CNN Training Script
# ENMGT 5400
#
# Requirements:
#   pip install tensorflow opencv-python numpy scikit-learn
#
# Usage:
#   python train.py
#
# Outputs:
#   model.h5        — saved Keras model (for TFLite conversion)
#   model_info.txt  — config + final accuracy summary

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET_DIR    = 'dataset'          # folder with rock/ paper/ scissors/ subfolders
CLASSES        = ['rock', 'paper', 'scissors']

# Preprocessing mode: 'rgb' | 'grayscale' | 'sobel'
#   rgb       — raw color, 32x32x3 input
#   grayscale — convert to gray, 32x32x1 input
#   sobel     — grayscale → Sobel edge map, 32x32x1 input (best for gesture shape)
PREPROCESSING  = 'sobel'

# Augmentation
AUG_FACTOR     = 10      # augmented samples generated per real training image

# Model size: 'standard' (32/64/128 filters, Dense 128) — high accuracy, slow on ESP32
#             'tiny'     (4/8/16 filters,  Dense  32) — ~50x faster on ESP32, ~3-5s/frame
MODEL_SIZE     = 'tiny'

# Model hyperparameters
EPOCHS         = 80
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
DROPOUT        = 0.4

RANDOM_SEED    = 42
# ──────────────────────────────────────────────────────────────────────────────


def apply_preprocessing(img_bgr):
    """
    Apply the selected preprocessing to a 32x32 BGR image.
    Returns a float32 array normalized to [0, 1].
    """
    if PREPROCESSING == 'rgb':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        return img  # shape: (32, 32, 3)

    elif PREPROCESSING == 'grayscale':
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
        return gray[:, :, np.newaxis]  # shape: (32, 32, 1)

    elif PREPROCESSING == 'sobel':
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype('float32')
        # Sobel in x and y directions, then magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        # Normalize to [0, 1]
        mag_max = mag.max()
        if mag_max > 0:
            mag /= mag_max
        return mag[:, :, np.newaxis]  # shape: (32, 32, 1)

    else:
        raise ValueError("Unknown PREPROCESSING mode: {}".format(PREPROCESSING))


def load_dataset():
    """Load all images from dataset/{rock,paper,scissors}/ and apply preprocessing."""
    images, labels = [], []
    counts = {}

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.bmp')])
        counts[cls] = len(files)

        for fname in files:
            img_bgr = cv2.imread(os.path.join(cls_dir, fname))
            if img_bgr is None:
                print("  WARNING: could not read", fname)
                continue
            images.append(apply_preprocessing(img_bgr))
            labels.append(label_idx)

    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int32')

    print("Loaded dataset:")
    for cls, n in counts.items():
        print("  {:10s}: {} images".format(cls, n))
    print("  Total     : {} images".format(len(y)))
    print("  Input shape: {}".format(X.shape[1:]))
    print("  Preprocessing: {}".format(PREPROCESSING))

    return X, y


def augment_training_set(X_train, y_train):
    """
    Use Keras ImageDataGenerator to create AUG_FACTOR augmented copies
    of each training image. Returns the expanded arrays.
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.7, 1.3],
        zoom_range=0.15,
    )

    aug_images, aug_labels = [X_train.copy()], [y_train.copy()]

    for _ in range(AUG_FACTOR):
        batch_aug = np.zeros_like(X_train)
        for i in range(len(X_train)):
            sample = X_train[i:i+1]  # keep 4D shape
            aug_iter = datagen.flow(sample, batch_size=1, seed=None)
            batch_aug[i] = next(aug_iter)[0]
        aug_images.append(batch_aug)
        aug_labels.append(y_train.copy())

    X_aug = np.concatenate(aug_images, axis=0)
    y_aug = np.concatenate(aug_labels, axis=0)

    # Shuffle
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


def build_model(input_shape, num_classes=3):
    """
    CNN for 32x32 input. Two sizes:
      standard: 32/64/128 filters, Dense 128 — high accuracy, slow on ESP32
      tiny:      4/8/16  filters, Dense  32 — ~50x faster on ESP32 (~3-5s/frame)
    """
    if MODEL_SIZE == 'tiny':
        f1, f2, f3, d1 = 4, 8, 16, 32
        model_name = 'rps_cnn_tiny'
    else:
        f1, f2, f3, d1 = 32, 64, 128, 128
        model_name = 'rps_cnn'

    model = keras.Sequential([
        # Block 1: 32x32 → 16x16
        layers.Conv2D(f1, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 2: 16x16 → 8x8
        layers.Conv2D(f2, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3: 8x8 → 4x4
        layers.Conv2D(f3, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Head
        layers.Flatten(),
        layers.Dense(d1, activation='relu'),
        layers.Dropout(DROPOUT),
        layers.Dense(num_classes, activation='softmax'),
    ], name=model_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join("{:>9}".format(c) for c in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = "  {:>7}".format(cls) + "  ".join("{:>9}".format(cm[i, j]) for j in range(len(CLASSES)))
        print(row)


def save_model_info(val_acc, history):
    lines = [
        "Rock Paper Scissors CNN — Training Summary",
        "=" * 45,
        "",
        "Config:",
        "  PREPROCESSING : {}".format(PREPROCESSING),
        "  EPOCHS        : {}".format(EPOCHS),
        "  BATCH_SIZE    : {}".format(BATCH_SIZE),
        "  LEARNING_RATE : {}".format(LEARNING_RATE),
        "  DROPOUT       : {}".format(DROPOUT),
        "  AUG_FACTOR    : {}".format(AUG_FACTOR),
        "",
        "Results:",
        "  Final val accuracy : {:.1f}%".format(val_acc * 100),
        "  Best val accuracy  : {:.1f}%".format(max(history.history['val_accuracy']) * 100),
        "  Epochs trained     : {}".format(len(history.history['loss'])),
        "",
        "Model saved to: model.h5",
    ]
    with open('model_info.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("\nSaved model_info.txt")


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1. Load data
    X, y = load_dataset()

    # 2. Train/val split (stratified 80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print("\nSplit: {} train, {} val".format(len(X_train), len(X_val)))

    # 3. Augment training set
    print("\nAugmenting training set (factor={})...".format(AUG_FACTOR))
    X_train_aug, y_train_aug = augment_training_set(X_train, y_train)
    print("  Augmented training size: {}".format(len(X_train_aug)))

    # 4. Build model
    input_shape = X.shape[1:]
    model = build_model(input_shape)
    model.summary()

    # 5. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=7,
            min_lr=1e-5, verbose=1
        ),
    ]

    # 6. Train
    print("\nTraining...")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print("\n" + "=" * 45)
    print("Final val accuracy : {:.1f}%".format(val_acc * 100))
    print("Best val accuracy  : {:.1f}%".format(max(history.history['val_accuracy']) * 100))

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print_confusion_matrix(y_val, y_pred)
    print("\nPer-class report:")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    # 8. Save
    model.save('model.h5')
    print("Saved model.h5")
    save_model_info(val_acc, history)

    # Rubric check
    if val_acc >= 0.75:
        print("\nBonus target met: {:.1f}% >= 75%".format(val_acc * 100))
    elif val_acc >= 0.50:
        print("\nAccuracy target met: {:.1f}% >= 50%".format(val_acc * 100))
    else:
        print("\nAccuracy below 50% — try adjusting hyperparameters or collecting more data.")


if __name__ == '__main__':
    main()
