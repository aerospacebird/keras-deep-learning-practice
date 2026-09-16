from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(5,(2,2), input_shape=(5,5,1))) # 5: kernnel size, (2,2) filter size, (5:height, 5:width, 1:channel)
model.add(Conv2D(3, (2,2)))

model.summary()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 8, 8, 10)            │             100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 7, 7, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 305 (1.19 KB)
#  Trainable params: 305 (1.19 KB)
#  Non-trainable params: 0 (0.00 B)

######################################################################################################################

# # Import the Sequential model from TensorFlow Keras
# from tensorflow.keras.models import Sequential

# # Import Dense and Conv2D layers from TensorFlow Keras
# from tensorflow.keras.layers import Dense, Conv2D


# # Create a Sequential model
# # Layers are added one after another in a simple linear sequence
# model = Sequential()


# # Add the first convolutional layer
# #
# # filters=10:
# #   The layer learns 10 different convolution filters (kernels).
# #
# # kernel_size=(3,3):
# #   Each filter has a 3x3 size.
# #
# # input_shape=(10,10,1):
# #   Input image size is 10x10 pixels with 1 channel (grayscale).
# #
# # Output shape:
# #   10x10 input - 3x3 kernel + 1 = 8x8
# #   Number of channels = 10
# #   Therefore, output shape = (8, 8, 10)
# #
# # No activation function is specified,
# # so the default activation is linear.
# model.add(
#     Conv2D(
#         10,
#         (3, 3),
#         input_shape=(10, 10, 1)
#     )
# )


# # Add the second convolutional layer
# #
# # filters=5:
# #   The layer learns 5 different convolution filters.
# #
# # kernel_size=(2,2):
# #   Each filter has a 2x2 size.
# #
# # Input from the previous layer:
# #   (8, 8, 10)
# #
# # Since the default padding is "valid":
# #   8x8 feature map - 2x2 kernel + 1 = 7x7
# #
# # Output shape:
# #   (7, 7, 5)
# #
# # The 10 input channels are automatically included in
# # the convolution operation. Each of the 5 filters has
# # a shape of (2, 2, 10).
# model.add(
#     Conv2D(
#         5,
#         (2, 2)
#     )
# )


# # Display the architecture of the model
# #
# # model.summary() shows:
# #   - Layer name
# #   - Layer type
# #   - Output shape
# #   - Number of trainable parameters
# model.summary()
# ```

# ### Expected output shape

# ```text
# Input
# (10, 10, 1)
#       ↓
# Conv2D(filters=10, kernel_size=(3,3))
#       ↓
# (8, 8, 10)
#       ↓
# Conv2D(filters=5, kernel_size=(2,2))
#       ↓
# (7, 7, 5)
# ```

# **Important:** `Dense` is imported in the code, but it is not used in this model. Therefore, you can simplify the import to:

# ```python
# from tensorflow.keras.layers import Conv2D
# ```
######################################################################################################################################