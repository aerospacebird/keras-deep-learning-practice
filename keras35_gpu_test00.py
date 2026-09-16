import tensorflow as tf
print(tf.__version__)

gpus= tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


if(gpus):
    print('GPU 있다.!')

else:
    print('GPU 없다.!!!')


#2.9.3
#PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
#GPU 있다.!    


##############################################################################################################

# 1. NVIDIA DRIVE 설치

# 2. CUDA 설치: 11.2.2

# 3. CUDNN 설치: 8.1.1 for 11.0, 11.1, 11.2  #  설치 파일을 풀고 해당 폴더의 파일을 cuda file의 폴더 파일에 (overwrite)에 덮어쓴다.

# 4. 가상환경 추가 (tf29x-gpu)

# 5. python 3.10

# 6. tensorflow 2.9.100(여러 모델이 시현되며, 그중에 택일한다.)

################################################################################################################