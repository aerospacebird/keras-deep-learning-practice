from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path = 'c:/study/_data/image/'

img = load_img(path + 'my_photo.png', target_size=(150,150))
#<PIL.Image.Image image mode=RGB size=150x150 at 0x2289A4019C0>
print(type(img))#<class 'PIL.Image.Image'>

# plt.imsfrom tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/'

img = load_img(path + 'my_photo.png', target_size=(150,150))
#<PIL.Image.Image image mode=RGB size=150x150 at 0x2289A4019C0>
print(type(img))#<class 'PIL.Image.Image'>

# plt.imshow(img)
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





