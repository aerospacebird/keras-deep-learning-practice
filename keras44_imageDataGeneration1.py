
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale=1./255,   #. 부동소숫점 형변환
    horizontal_flip=True,  #이미지를 좌우 반전합니다.
    vertical_flip=True,    #이미지를 상하 반전합니다.
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7, # 전단 변환(Shear Transformation)**을 적용하는 옵션
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False,
)
#Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False,
)
#Found 120 images belonging to 2 classes.


print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x0000025188F379D0>
print(xy_test)
#<keras.preprocessing.image.DirectoryIterator object at 0x0000025188F379D0>

print(xy_train.next()) # Iterator 첫번째를 보여줘?
print(xy_train.next()) # 두번째 Iterator 출력해줘?

# print(xy_train[0])
# print(xy_train[1])
# print(xy_train[2])

#print(xy_train[0][0]  # 첫번째 배치의 X 데이터가 되겠지요.
#print(xy_train[0][1]  # 첫번째 배치의 Y 데이터가 되겠지요.
print(xy_train[0][0].shape) # (10, 100, 100, 1)
print(xy_train[0][1].shape) #(10,)

#print(xy_train)[16][0]) # 여기서 부터 에러, 이유는 160장이다..? 배치는 10개이니까.

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

