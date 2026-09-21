from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/'

img = load_img(path + '내사진.png', target_size=(150,150))
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

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + "keras48_me.npy", arr=arr)





# np_path = './_data/kaggle_cat_dog_npy/'             #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_x_train.npy' , arr = xy_train[0][0])  #또는 arr = x_train 도가능 
# np.save(np_path + 'keras45_01_y_train.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_x_test.npy' , arr = xy_train[0][0])  #🤎💛🧡 🤎💛🧡
# np.save(np_path + 'keras45_01_y_test.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡





