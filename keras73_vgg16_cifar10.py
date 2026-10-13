

# ============================================================
# CRAZY AI VGG16 TRANSFER LEARNING V1.0
# TensorFlow / Keras + CIFAR-10
# ============================================================

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar10

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
SEED = 333
IMAGE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 10

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

OUTPUT_DIR = "/content/crazy_ai_vgg16"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. Load CIFAR-10
# ------------------------------------------------------------
(x_all, y_all), (x_test, y_test) = cifar10.load_data()

y_all = y_all.reshape(-1).astype("int32")
y_test = y_test.reshape(-1).astype("int32")

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Original training images:", x_all.shape)
print("Test images:", x_test.shape)
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

# ------------------------------------------------------------
# 4. Image preprocessing
# ------------------------------------------------------------
# VGG16 preprocessing expects RGB images in the 0-255 range.
# It converts RGB to BGR and applies ImageNet channel centering.
# Do not divide images by 255 before preprocess_input.

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = preprocess_input(image)
    return image, tf.cast(label, tf.int32)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000, seed=SEED)
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ------------------------------------------------------------
# 5. Load pretrained VGG16
# ------------------------------------------------------------
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze pretrained convolutional layers.
base_model.trainable = False

print("\nBackbone layers:", len(base_model.layers))
print("Backbone trainable:", base_model.trainable)

# ------------------------------------------------------------
# 6. Build a new CIFAR-10 classifier
# ------------------------------------------------------------
inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

features = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(features)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs,
    name="CRAZY_AI_VGG16_CIFAR10"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# 7. Verify parameter counts
# ------------------------------------------------------------
total_params = model.count_params()

trainable_params = sum(
    np.prod(w.shape)
    for w in model.trainable_weights
)

non_trainable_params = total_params - trainable_params

print("\n========== PARAMETER REPORT ==========")
print(f"Total parameters: {int(total_params):,}")
print(f"Trainable parameters: {int(trainable_params):,}")
print(f"Non-trainable parameters: {int(non_trainable_params):,}")

# ------------------------------------------------------------
# 8. Callbacks
# ------------------------------------------------------------
best_model_path = os.path.join(
    OUTPUT_DIR, "vgg16_cifar10_best.keras"
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
# 9. Train
# ------------------------------------------------------------
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

# Restore best validation checkpoint.
model = tf.keras.models.load_model(best_model_path)

# ------------------------------------------------------------
# 10. Final test evaluation
# ------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

y_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

print("\n========== TEST RESULTS ==========")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
))

# ------------------------------------------------------------
# 11. Plot loss and accuracy
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
axes[0].set_title("VGG16 Loss")
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
axes[1].set_title("VGG16 Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "training_curves.png"),
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 12. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

plt.title("VGG16 CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 13. Save final model
# ------------------------------------------------------------
final_model_path = os.path.join(
    OUTPUT_DIR, "vgg16_cifar10_final.keras"
)

model.save(final_model_path)

print("\n========== FINAL REPORT ==========")
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Total parameters: {int(total_params):,}")
print(f"Trainable parameters: {int(trainable_params):,}")
print(f"Non-trainable parameters: {int(non_trainable_params):,}")
print("Output directory:", OUTPUT_DIR)







# import numpy as np
# #################################################################################################################
# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# import tensorflow as tf

# random.seed(333)
# np.random.seed(333)
# tf.random.set_seed(333)

# from keras.applications import VGG16
# from tensorflow.keras.datasets import cifar10

# #2. MODEL
# vgg16 = VGG16()
# ##########  디폴트  ############################
# model = VGG16(
#     weights='imagenet',
#     include_top=True,       #True,
#     input_shape=(224, 224,3),
# )


# vgg16.trainable=False  # 가중치 동결시켜놓는다.

# #model.summary()
# model = Sequential()

# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(10, activation='softmax'))

# model.summary()


# exit()

# #실습#
# #비교할것.
# #1. 이전에 본인이 한 최상의 결과,
# #2. 가중치 동결하고, 훈현시켰을때, trainable=False
# #3. 가중치 동결하지 않고 훈련했을때, trainable=True
# #4. 시간까지 비교.



# # model.summary()
# # Model: "vgg16"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #   
# # =================================================================
# #  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
# #  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
# #  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
# #  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
# #  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
# #  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
# #  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
# #  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
# #  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
# #  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
# #  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
# #  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
# #  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
# #  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
# #  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
# #  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
# #  flatten (Flatten)           (None, 25088)             0         
                                                                 
# #  fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
# #  fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
# #  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# # =================================================================
# # Total params: 138,357,544
# # Trainable params: 138,357,544
# # Non-trainable params: 0
# # _________________________________________________________________
# # PS C:\study> 
# ######################################################################################################

# # Model: "vgg16"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #   
# # =================================================================
# #  input_1 (InputLayer)        [(None, 100, 100, 3)]     0         
                                                                 
# #  block1_conv1 (Conv2D)       (None, 100, 100, 64)      1792      
                                                                 
# #  block1_conv2 (Conv2D)       (None, 100, 100, 64)      36928     
                                                                 
# #  block1_pool (MaxPooling2D)  (None, 50, 50, 64)        0         
                                                                 
# #  block2_conv1 (Conv2D)       (None, 50, 50, 128)       73856     
                                                                 
# #  block2_conv2 (Conv2D)       (None, 50, 50, 128)       147584    
                                                                 
# #  block2_pool (MaxPooling2D)  (None, 25, 25, 128)       0         
                                                                 
# #  block3_conv1 (Conv2D)       (None, 25, 25, 256)       295168    
                                                                 
# #  block3_conv2 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
# #  block3_conv3 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
# #  block3_pool (MaxPooling2D)  (None, 12, 12, 256)       0         
                                                                 
# #  block4_conv1 (Conv2D)       (None, 12, 12, 512)       1180160   
                                                                 
# #  block4_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
# #  block4_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
# #  block4_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
# #  block5_conv1 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
# #  block5_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
# #  block5_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
# #  block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
# # =================================================================
# # Total params: 14,714,688
# # Trainable params: 14,714,688
# # Non-trainable params: 0
# # _________________________________________________________________
# # PS C:\study> ^C
# # PS C:\study> 
# # PS C:\study>  c:; cd 'c:\study'; & 'C:\Users\Admin\miniconda3\envs\tf30x-gpu\python.exe' 'c:\Users\Admin\.vscode\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '14203' '--' 'c:\study\keras2\keras72_1_vgg16_include_top.py' 
# # 2026-10-13 13:46:56.462755: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# # To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # 2026-10-13 13:46:56.883166: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5456 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
# # Model: "vgg16"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #   
# # =================================================================
# #  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
# #  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
# #  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
# #  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
# #  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
# #  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
# #  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
# #  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
# #  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
# #  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
# #  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
# #  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
# #  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
# #  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
# #  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
# #  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
# #  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
# #  flatten (Flatten)           (None, 25088)             0         
                                                                 
# #  fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
# #  fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
# #  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# # =================================================================
# # Total params: 138,357,544
# # Trainable params: 138,357,544
# # Non-trainable params: 0
# # _____________________________________________
# ###########################################################################################################
# model = VGG16(
#     weights='imagenet',
#     include_top=False,#추출된 이미지 특징을 활용해 개와 고양이용 분류기를 새로 구성할 수 있습니다.
#     input_shape=(100, 100, 3),
# )
# model.summary()