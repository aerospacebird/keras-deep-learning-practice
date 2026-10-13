
# ============================================================
# CRAZY AI VGG16 TRANSFER LEARNING V1.2
# TensorFlow / Keras + CIFAR-10
#
# Architecture:
# VGG16(include_top=False)
# -> Flatten
# -> Dense(100, ReLU)
# -> Dense(100, ReLU)
# -> Dense(10, Softmax)
#
# Compatible with TensorFlow 2.9.3
# Windows / Google Colab
# ============================================================

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar10

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
SEED = 333
IMAGE_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 10
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Windows / Google Colab output directory
if os.path.isdir("/content"):
    OUTPUT_DIR = "/content/crazy_ai_vgg16"
else:
    OUTPUT_DIR = os.path.abspath("crazy_ai_vgg16")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure GPU memory growth before GPU initialization
gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print("GPU memory configuration:", error)

print("=" * 65)
print("CRAZY AI VGG16 TRANSFER LEARNING V1.2")
print("=" * 65)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Output directory:", OUTPUT_DIR)

# ------------------------------------------------------------
# 2. Load CIFAR-10 dataset
# ------------------------------------------------------------
(x_all, y_all), (x_test, y_test) = cifar10.load_data()

# Convert labels from (N, 1) to (N,)
y_all = y_all.squeeze().astype(np.int32)
y_test = y_test.squeeze().astype(np.int32)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\n========== DATASET REPORT ==========")
print("Original training images:", x_all.shape)
print("Training labels:", y_all.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)
print("Number of classes:", len(class_names))

# ------------------------------------------------------------
# 3. Deterministic train / validation split
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(x_all))

train_idx = indices[:45000]
val_idx = indices[45000:]

x_train = x_all[train_idx]
y_train = y_all[train_idx]

x_val = x_all[val_idx]
y_val = y_all[val_idx]

print("\nTraining samples:", len(x_train))
print("Validation samples:", len(x_val))
print("Test samples:", len(x_test))

# ------------------------------------------------------------
# 4. VGG16 image preprocessing
# ------------------------------------------------------------
# Input images are RGB uint8 values in [0, 255].
# Resize first, then apply VGG16 ImageNet preprocessing.
# Do NOT divide by 255 before preprocess_input.

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    image = preprocess_input(image)
    label = tf.cast(label, tf.int32)

    return image, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(
        buffer_size=10000,
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ------------------------------------------------------------
# 5. Load pretrained VGG16 backbone ONCE
# ------------------------------------------------------------
print("\n========== LOADING VGG16 ==========")

vgg16 = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Stage 1: freeze all pretrained VGG16 layers
vgg16.trainable = False

print("VGG16 input shape:", vgg16.input_shape)
print("VGG16 output shape:", vgg16.output_shape)
print("VGG16 layers:", len(vgg16.layers))
print("VGG16 trainable:", vgg16.trainable)
print("VGG16 parameters:", f"{vgg16.count_params():,}")

# ------------------------------------------------------------
# 6. Build requested Sequential architecture
# ------------------------------------------------------------
# IMPORTANT:
# include_top=False ensures VGG16 returns feature maps,
# not the original 1000-class ImageNet predictions.

model = Sequential(
    name="CRAZY_AI_VGG16_CIFAR10"
)

model.add(vgg16)
model.add(Flatten(name="flatten"))
model.add(Dense(100, activation="relu", name="dense_100_1"))
model.add(Dense(100, activation="relu", name="dense_100_2"))
model.add(
    Dense(
        NUM_CLASSES,
        activation="softmax",
        name="cifar10_output"
    )
)

# Build model and print architecture
model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

print("\n========== MODEL SUMMARY ==========")
model.summary()

# Verify the backbone output tensor shape
print("\nVGG16 feature map shape:", vgg16.output_shape)
print("Model output shape:", model.output_shape)

assert model.output_shape[-1] == NUM_CLASSES
assert vgg16.output_shape[-1] == 512

# ------------------------------------------------------------
# 7. Compile model
# ------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------
# 8. Verify parameter counts
# ------------------------------------------------------------
total_params = model.count_params()

trainable_params = sum(
    int(np.prod(w.shape))
    for w in model.trainable_weights
)

non_trainable_params = sum(
    int(np.prod(w.shape))
    for w in model.non_trainable_weights
)

print("\n========== PARAMETER REPORT ==========")
print(f"Total parameters:       {total_params:,}")
print(f"Trainable parameters:   {trainable_params:,}")
print(f"Non-trainable params:   {non_trainable_params:,}")

assert total_params == trainable_params + non_trainable_params

# Expected with IMAGE_SIZE=100:
# Total:           15,186,698
# Trainable:          472,010
# Non-trainable:   14,714,688

# ------------------------------------------------------------
# 9. Callbacks
# ------------------------------------------------------------
best_model_path = os.path.join(
    OUTPUT_DIR,
    "vgg16_cifar10_best.keras"
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
]

# ------------------------------------------------------------
# 10. Train model
# ------------------------------------------------------------
print("\n========== TRAINING START ==========")

start_time = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_seconds = time.perf_counter() - start_time

print(f"\nTraining time: {training_seconds:.2f} seconds")

# Load best checkpoint selected by validation accuracy
model = tf.keras.models.load_model(best_model_path)

# ------------------------------------------------------------
# 11. Evaluate on held-out CIFAR-10 test set
# ------------------------------------------------------------
print("\n========== TEST EVALUATION ==========")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

y_prob = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)

print("\n========== TEST RESULTS ==========")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 12. Plot training and validation curves
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    history.history["loss"],
    label="Train Loss"
)

axes[0].plot(
    history.history["val_loss"],
    label="Validation Loss"
)

axes[0].set_title("CRAZY AI VGG16 Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

axes[1].plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

axes[1].set_title("CRAZY AI VGG16 Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

curve_path = os.path.join(
    OUTPUT_DIR,
    "training_curves.png"
)

plt.savefig(
    curve_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ------------------------------------------------------------
# 13. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

plt.title("CRAZY AI VGG16 CIFAR-10 Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ------------------------------------------------------------
# 14. Save final model
# ------------------------------------------------------------
final_model_path = os.path.join(
    OUTPUT_DIR,
    "vgg16_cifar10_final.keras"
)

model.save(final_model_path)

# ------------------------------------------------------------
# 15. Final report
# ------------------------------------------------------------
print("\n" + "=" * 65)
print("CRAZY AI VGG16 TRANSFER LEARNING V1.2")
print("========== FINAL REPORT ==========")
print("=" * 65)

print("Best model:", best_model_path)
print("Final model:", final_model_path)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print(f"Total parameters: {model.count_params():,}")

print(
    "Trainable parameters:",
    f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}"
)

print(
    "Non-trainable parameters:",
    f"{sum(int(np.prod(w.shape)) for w in model.non_trainable_weights):,}"
)

print(f"Training time: {training_seconds:.2f} seconds")
print("Training curves:", curve_path)
print("Confusion matrix:", cm_path)
print("Output directory:", OUTPUT_DIR)

print("\nCRAZY AI VGG16 TRANSFER LEARNING COMPLETED.")
###############################################################################################################
# ========== TEST EVALUATION ==========
# 625/625 [==============================] - 9s 14ms/step - loss: 0.5596 - accuracy: 0.8431
# 625/625 [==============================] - 9s 14ms/step

# ========== TEST RESULTS ==========
# Test loss: 0.5596
# Test accuracy: 0.8431

# Classification report:
#               precision    recall  f1-score   support

#     airplane     0.8661    0.8800    0.8730      1000
#   automobile     0.9288    0.9000    0.9142      1000
#         bird     0.8442    0.7800    0.8108      1000
#          cat     0.7247    0.7080    0.7162      1000
#         deer     0.7281    0.8810    0.7973      1000
#          dog     0.7826    0.7630    0.7727      1000
#         frog     0.8678    0.8730    0.8704      1000
#        horse     0.9207    0.8240    0.8697      1000
#         ship     0.9269    0.8880    0.9070      1000
#        truck     0.8729    0.9340    0.9024      1000

#     accuracy                         0.8431     10000
#    macro avg     0.8463    0.8431    0.8434     10000
# weighted avg     0.8463    0.8431    0.8434     10000


# =================================================================
# CRAZY AI VGG16 TRANSFER LEARNING V1.2
# ========== FINAL REPORT ==========
# =================================================================
# Best model: /content/crazy_ai_vgg16\vgg16_cifar10_best.keras
# Final model: /content/crazy_ai_vgg16\vgg16_cifar10_final.keras
# Test loss: 0.5596
# Test accuracy: 0.8431
# Total parameters: 15,186,698
# Trainable parameters: 472,010
# Non-trainable parameters: 14,714,688
# Training time: 216.35 seconds
# Training curves: /content/crazy_ai_vgg16\training_curves.png
# Confusion matrix: /content/crazy_ai_vgg16\confusion_matrix.png
# Output directory: /content/crazy_ai_vgg16

# CRAZY AI VGG16 TRANSFER LEARNING COMPLETED.

