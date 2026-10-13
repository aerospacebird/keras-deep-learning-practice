
# VGG16 모델 (Visual Geometry Group 16)
# VGG16은 이미지 분류와 컴퓨터 비전 분야에서 널리 알려진 CNN(Convolutional Neural Network) 모델입니다. 
# 영국 옥스퍼드대학교의 Visual Geometry Group(VGG)이 개발했으며, 
# 2014년 ILSVRC(ImageNet Large Scale Visual Recognition Challenge)에서 뛰어난 성능을 보였습니다.

# VGG16의 이름에서 16은 학습 가능한 가중치를 가진 층이 16개라는 의미입니다.
# - 컨볼루션 층: 13개
# - 완전연결층(Fully Connected): 3개
# - 최대 풀링(Max Pooling): 5개
# - 활성화 함수: 일반적으로 ReLU
# - 대표 입력 크기: \(224\times224\times3\)
# - ImageNet 사전 학습 모델의 출력: 1,000개 클래스

#################################################################################################################
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16

#2. MODEL
#model = VGG16()
###########  디폴트  ############################
# model = VGG16(
#     weights='imagenet',
#     include_top=True,       #True,
#     input_shape=(224, 224, 3),
# )
# model.summary()

# model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
#  flatten (Flatten)           (None, 25088)             0         
                                                                 
#  fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
#  fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
#  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\study> 
######################################################################################################

# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 100, 100, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 100, 100, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 100, 100, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 50, 50, 64)        0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 50, 50, 128)       73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 50, 50, 128)       147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 25, 25, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 25, 25, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 12, 12, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 12, 12, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\study> ^C
# PS C:\study> 
# PS C:\study>  c:; cd 'c:\study'; & 'C:\Users\Admin\miniconda3\envs\tf30x-gpu\python.exe' 'c:\Users\Admin\.vscode\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '14203' '--' 'c:\study\keras2\keras72_1_vgg16_include_top.py' 
# 2026-10-13 13:46:56.462755: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2026-10-13 13:46:56.883166: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5456 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
#  flatten (Flatten)           (None, 25088)             0         
                                                                 
#  fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
#  fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
#  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _____________________________________________
###########################################################################################################
model = VGG16(
    weights='imagenet',
    include_top=False,#추출된 이미지 특징을 활용해 개와 고양이용 분류기를 새로 구성할 수 있습니다.
    input_shape=(100, 100, 3),
)
model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 100, 100, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 100, 100, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 100, 100, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 50, 50, 64)        0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 50, 50, 128)       73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 50, 50, 128)       147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 25, 25, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 25, 25, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 12, 12, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 12, 12, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
################### include_top = False  ##########################################
#1. FC Layer가 없어짐.(직접 아래에 fc layer를 명시해주면됨)
#2. input_shape를 원하는 데이터 shape로 변경.
###################################################################################3