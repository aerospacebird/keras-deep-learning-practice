from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# path = 'c:/study/_data/image/'

# img = load_img(path + 'my_photo.png', target_size=(150,150))
# #<PIL.Image.Image image mode=RGB size=150x150 at 0x2289A4019C0>
# print(type(img))#<class 'PIL.Image.Image'>

# # plt.imsfrom tensorflow.keras.preprocessing.image import load_img 
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt

# path = 'c:/study/_data/image/'

# img = load_img(path + 'my_photo.png', target_size=(150,150))
# #<PIL.Image.Image image mode=RGB size=150x150 at 0x2289A4019C0>
# print(type(img))#<class 'PIL.Image.Image'>

# # plt.imshow(img)
# # plt.show()

# arr = img_to_array(img)
# print(arr)
# print(arr.shape)    #(150, 150, 3)
# print(type(arr))    #<class 'numpy.ndarray'>

# arr = np.expand_dims(arr, axis=0) #차원 증가 
# # print(arr)
# print(arr.shape)    #(1, 150, 150, 3)

# # np_path = './_data/kaggle_cat_dog_npy/'
# # np.save(np_path + "keras48_me.npy", arr=arr)

################# 요기부터 증폭이닷 ####################
datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=  True,    #수평 뒤집기, 
    vertical_flip= True,        #수직 뒤집기, 
    width_shift_range= 0.1,     #평형이동,
    height_shift_range=0.1,
    rotation_range= 15,           #각도조절(정해진 각도만큼 이미지 회전)
    zoom_range= 1.2,
    shear_range= 0.7,           #좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest'         
)

augment_size = 100

print(x_train.shape) #(60000, 28, 28)
print(x_train[0].shape)# (28, 28)

aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1) # 복사하여 붙이기
#exit()
print(aaa.shape) #(28, 28), (28, 2800)------------reshape 하면 요렇게 된다.------------->>>>>>>>>>>>>> (100, 28, 28, 1)

xy_data = datagen.flow(
      np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),  # x       28*28 = 784
      np.zeros(augment_size), # y
      batch_size= augment_size,
      shuffle=False,

).next()

print(xy_data)
print(type(xy_data))

#print(xy_data.shape)
print(len(xy_data)) # x, y 두개니까

print(xy_data[0].shape) #(100, 28, 28, 1)
print(xy_data[1].shape) # (100,)


plt.figure(figsize=(7,7))
for i in range(49):
      plt.subplot(7,7,i+1)
      plt.imshow(xy_data[0][i], cmap='gray')
plt.show()


      
exit()







it = datagen.flow(arr, 
             batch_size=1,
)
print(it)
#<keras.preprocessing.image.NumpyArrayIterator object at 0x00000171C9413F70>

# print(it.next())      #파이썬3.10까지,
print(next(it))         #파이썬3.11이후
print(next(it).shape)   #(1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(150,150,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()


how(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    #(150, 150, 3)
print(type(arr))    #<class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) #차원 증가 
# print(arr)
print(arr.shape)    #(1, 150, 150, 3)

# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + "keras48_me.npy", arr=arr)

############################################### 요기서 부터 증폭입니다.######################################
datagen = ImageDataGenerator(
    rescale=1./255,   #. 부동소숫점 형변환
    horizontal_flip=True,  #이미지를 좌우 반전합니다.
    vertical_flip=True,    #이미지를 상하 반전합니다.
    width_shift_range=0.9,
    height_shift_range=0.9,
    rotation_range=35,
    zoom_range=6,
    shear_range=0.7, # 전단 변환(Shear Transformation)**을 적용하는 옵션
    fill_mode='nearest'
)

it = datagen.flow(arr,
              batch_size=1,

)
print(it) #python 3.10

print(it.next())#pthon 3.11이후

print(next(it).shape) #(1, 150, 150, 3)

arr= img_to_array(img)
print(arr)


plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in  range(5):  # 0~4까지

        batch = next(it)
        #print(batch.shape) (1, 150, 150, 3)
        batch = batch.reshape(150,150,3) # 값과 순서가 바뀌면 않된다. reshape해도

        ax[i].imshow(batch)
plt.show()







# np_path = './_data/kaggle_cat_dog_npy/'             #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_x_train.npy' , arr = xy_train[0][0])  #또는 arr = x_train 도가능 
# np.save(np_path + 'keras45_01_y_train.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_x_test.npy' , arr = xy_train[0][0])  #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_y_test.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡


# # ============================================================
# # 1. 필요한 라이브러리 import
# # ============================================================
# from tensorflow.keras.preprocessing.image import load_img      # 이미지 파일을 불러오는 함수
# from tensorflow.keras.preprocessing.image import img_to_array  # PIL 이미지를 numpy 배열로 변환하는 함수
# from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 이미지 데이터 증강(Augmentation)을 수행하는 클래스
# import numpy as np                                             # 수치 계산 및 배열 연산을 위한 라이브러리
# import matplotlib.pyplot as plt                                # 이미지를 화면에 시각화하기 위한 라이브러리


# # ============================================================
# # 2. 이미지 경로 설정 및 이미지 불러오기
# # ============================================================
# path = 'c:/study/_data/image/'          # 이미지가 저장된 폴더 경로

# # load_img() : 이미지 파일을 불러와서 PIL.Image 객체로 반환
# # target_size=(150, 150) → 원본 이미지 크기에 상관없이 150x150으로 리사이즈
# img = load_img(path + 'my_photo.png', target_size=(150, 150))

# print(type(img))   # <class 'PIL.Image.Image'>  → PIL 이미지 객체임을 확인


# # ============================================================
# # 3. PIL 이미지를 numpy 배열로 변환
# # ============================================================
# arr = img_to_array(img)                # (높이, 너비, 채널) 형태의 numpy 배열로 변환
# print(arr)                             # 픽셀 값 출력 (0~255 범위의 float32)
# print(arr.shape)                       # (150, 150, 3)  → 높이 150, 너비 150, RGB 3채널
# print(type(arr))                       # <class 'numpy.ndarray'>


# # ============================================================
# # 4. 배치 차원 추가 (모델 입력 형식에 맞추기)
# # ============================================================
# # Keras의 대부분의 함수는 (batch_size, height, width, channels) 형태를 기대함
# # 현재 arr는 (150, 150, 3)이므로 맨 앞에 배치 차원을 추가해야 함
# arr = np.expand_dims(arr, axis=0)      # axis=0 → 0번째 축에 1 추가
# print(arr.shape)                       # (1, 150, 150, 3)  → 배치 크기 1


# # ============================================================
# # 5. 이미지 데이터 증강기(ImageDataGenerator) 설정
# # ============================================================
# # ImageDataGenerator는 원본 이미지를 실시간으로 변형(증강)하여
# # 학습 데이터의 다양성을 높여주는 역할을 함
# datagen = ImageDataGenerator(
#     rescale=1./255,            # 픽셀 값을 0~1 범위로 정규화 (원래 0~255 → 0.0~1.0)
#     horizontal_flip=True,      # 좌우 반전 (수평 뒤집기)
#     vertical_flip=True,        # 상하 반전 (수직 뒤집기)
#     width_shift_range=0.1,     # 가로 방향으로 최대 10% 이동
#     height_shift_range=0.1,    # 세로 방향으로 최대 10% 이동
#     rotation_range=15,         # ±15도 범위 내에서 랜덤 회전
#     zoom_range=1.2,            # 0.8~1.2배 범위로 확대/축소 (1.2는 최대 1.2배까지)
#     shear_range=0.7,           # 전단 변환(찌그러뜨리기) 강도
#     fill_mode='nearest'        # 빈 공간을 가장 가까운 픽셀 값으로 채움
# )


# # ============================================================
# # 6. 증강된 이미지를 생성하는 이터레이터(Iterator) 만들기
# # ============================================================
# # flow() : numpy 배열을 받아서 증강된 배치를 계속해서 생성해주는 이터레이터 반환
# it = datagen.flow(
#     arr,                       # 입력 데이터 (배치 차원이 있는 이미지)
#     batch_size=1               # 한 번에 1장씩 생성
# )

# print(it)                      # <keras...NumpyArrayIterator object ...> 출력


# # ============================================================
# # 7. 이터레이터에서 한 장씩 꺼내보기 (테스트)
# # ============================================================
# # next(it) → 이터레이터에서 다음 배치를 가져옴
# print(next(it))                # 증강된 이미지의 픽셀 값 출력
# print(next(it).shape)          # (1, 150, 150, 3)  → 배치 크기 1인 이미지


# # ============================================================
# # 8. 증강된 이미지 5장을 시각화
# # ============================================================
# # 1행 5열의 subplot을 생성
# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))

# for i in range(5):
#     batch = next(it)                   # 이터레이터에서 증강된 이미지 1장 가져오기
#     # batch.shape → (1, 150, 150, 3)
    
#     batch = batch.reshape(150, 150, 3) # 배치 차원을 제거하여 (150, 150, 3)으로 만듦
#                                        # (imshow는 3차원 배열을 기대함)
    
#     ax[i].imshow(batch)                # i번째 subplot에 이미지 표시
#     ax[i].axis('off')                  # 축(눈금) 숨기기

# plt.tight_layout()                     # subplot 간격 자동 조정
# plt.show()                             # 화면에 출력


