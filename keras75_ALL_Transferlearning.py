import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2 
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet121, DenseNet169
from keras.applications import InceptionResNetV2, InceptionV3
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB2
from keras.applications import Xception

model_list = [
        VGG16(),VGG19(), ResNet50(),ResNet50V2(),ResNet101(),ResNet101V2(),ResNet152(),ResNet152V2(),
        DenseNet121(),DenseNet169(),DenseNet201(),InceptionV3(),InceptionResNetV2(),MobileNet(),
        MobileNetV2(),MobileNetV3Small(),MobileNetV3Large(),NASNetMobile(), NASNetLarge(),
        EfficientNetB0(),EfficientNetB1(),EfficientNetB2()
]

model = Xception()

model.trainable = True

model.summary()

print("==================================================================")
print("모델명 :", model.name)
print("전체 가중치 갯수:", len(model.weights))
print("전체 훈련가능 가중치 갯수: ", len(model.trainable_weights))


for model in model_list:
        print("==================================================================")
        print("모델명 :", model.name)
        print("전체 가중치 갯수:", len(model.weights))
        print("전체 훈련가능 가중치 갯수: ", len(model.trainable_weights))
        
################################################################################################################
# Model: "xception"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_23 (InputLayer)          [(None, 299, 299, 3  0           []                               
#                                 )]                                                                
                                                                                                  
#  block1_conv1 (Conv2D)          (None, 149, 149, 32  864         ['input_23[0][0]']               
#                                 )                                                                 
                                                                                                  
#  block1_conv1_bn (BatchNormaliz  (None, 149, 149, 32  128        ['block1_conv1[0][0]']           
#  ation)                         )                                                                 
                                                                                                  
#  block1_conv1_act (Activation)  (None, 149, 149, 32  0           ['block1_conv1_bn[0][0]']        
#                                 )                                                                 
                                                                                                  
#  block1_conv2 (Conv2D)          (None, 147, 147, 64  18432       ['block1_conv1_act[0][0]']       
#                                 )                                                                 
                                                                                                  
#  block1_conv2_bn (BatchNormaliz  (None, 147, 147, 64  256        ['block1_conv2[0][0]']           
#  ation)                         )                                                                 
                                                                                                  
#  block1_conv2_act (Activation)  (None, 147, 147, 64  0           ['block1_conv2_bn[0][0]']        
#                                 )                                                                 
                                                                                                  
#  block2_sepconv1 (SeparableConv  (None, 147, 147, 12  8768       ['block1_conv2_act[0][0]']       
#  2D)                            8)                                                                
                                                                                                  
#  block2_sepconv1_bn (BatchNorma  (None, 147, 147, 12  512        ['block2_sepconv1[0][0]']        
#  lization)                      8)                                                                
                                                                                                  
#  block2_sepconv2_act (Activatio  (None, 147, 147, 12  0          ['block2_sepconv1_bn[0][0]']     
#  n)                             8)                                                                
                                                                                                  
#  block2_sepconv2 (SeparableConv  (None, 147, 147, 12  17536      ['block2_sepconv2_act[0][0]']    
#  2D)                            8)                                                                
                                                                                                  
#  block2_sepconv2_bn (BatchNorma  (None, 147, 147, 12  512        ['block2_sepconv2[0][0]']        
#  lization)                      8)                                                                
                                                                                                  
#  conv2d_297 (Conv2D)            (None, 74, 74, 128)  8192        ['block1_conv2_act[0][0]']       
                                                                                                  
#  block2_pool (MaxPooling2D)     (None, 74, 74, 128)  0           ['block2_sepconv2_bn[0][0]']     
                                                                                                  
#  batch_normalization_297 (Batch  (None, 74, 74, 128)  512        ['conv2d_297[0][0]']             
#  Normalization)                                                                                   
                                                                                                  
#  add_8 (Add)                    (None, 74, 74, 128)  0           ['block2_pool[0][0]',            
#                                                                   'batch_normalization_297[0][0]']
                                                                                                  
#  block3_sepconv1_act (Activatio  (None, 74, 74, 128)  0          ['add_8[0][0]']                  
#  n)                                                                                               
                                                                                                  
#  block3_sepconv1 (SeparableConv  (None, 74, 74, 256)  33920      ['block3_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block3_sepconv1_bn (BatchNorma  (None, 74, 74, 256)  1024       ['block3_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block3_sepconv2_act (Activatio  (None, 74, 74, 256)  0          ['block3_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block3_sepconv2 (SeparableConv  (None, 74, 74, 256)  67840      ['block3_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block3_sepconv2_bn (BatchNorma  (None, 74, 74, 256)  1024       ['block3_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  conv2d_298 (Conv2D)            (None, 37, 37, 256)  32768       ['add_8[0][0]']                  
                                                                                                  
#  block3_pool (MaxPooling2D)     (None, 37, 37, 256)  0           ['block3_sepconv2_bn[0][0]']     
                                                                                                  
#  batch_normalization_298 (Batch  (None, 37, 37, 256)  1024       ['conv2d_298[0][0]']             
#  Normalization)                                                                                   
                                                                                                  
#  add_9 (Add)                    (None, 37, 37, 256)  0           ['block3_pool[0][0]',            
#                                                                   'batch_normalization_298[0][0]']
                                                                                                  
#  block4_sepconv1_act (Activatio  (None, 37, 37, 256)  0          ['add_9[0][0]']                  
#  n)                                                                                               
                                                                                                  
#  block4_sepconv1 (SeparableConv  (None, 37, 37, 728)  188672     ['block4_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block4_sepconv1_bn (BatchNorma  (None, 37, 37, 728)  2912       ['block4_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block4_sepconv2_act (Activatio  (None, 37, 37, 728)  0          ['block4_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block4_sepconv2 (SeparableConv  (None, 37, 37, 728)  536536     ['block4_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block4_sepconv2_bn (BatchNorma  (None, 37, 37, 728)  2912       ['block4_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  conv2d_299 (Conv2D)            (None, 19, 19, 728)  186368      ['add_9[0][0]']                  
                                                                                                  
#  block4_pool (MaxPooling2D)     (None, 19, 19, 728)  0           ['block4_sepconv2_bn[0][0]']     
                                                                                                  
#  batch_normalization_299 (Batch  (None, 19, 19, 728)  2912       ['conv2d_299[0][0]']             
#  Normalization)                                                                                   
                                                                                                  
#  add_10 (Add)                   (None, 19, 19, 728)  0           ['block4_pool[0][0]',            
#                                                                   'batch_normalization_299[0][0]']
                                                                                                  
#  block5_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_10[0][0]']                 
#  n)                                                                                               
                                                                                                  
#  block5_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block5_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block5_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block5_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block5_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block5_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block5_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block5_sepconv2_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block5_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv3_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block5_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv3[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  add_11 (Add)                   (None, 19, 19, 728)  0           ['block5_sepconv3_bn[0][0]',     
#                                                                   'add_10[0][0]']                 
                                                                                                  
#  block6_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_11[0][0]']                 
#  n)                                                                                               
                                                                                                  
#  block6_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block6_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block6_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block6_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block6_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block6_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block6_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block6_sepconv2_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block6_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv3_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block6_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv3[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  add_12 (Add)                   (None, 19, 19, 728)  0           ['block6_sepconv3_bn[0][0]',     
#                                                                   'add_11[0][0]']                 
                                                                                                  
#  block7_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_12[0][0]']                 
#  n)                                                                                               
                                                                                                  
#  block7_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block7_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block7_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block7_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block7_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block7_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block7_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block7_sepconv2_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block7_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv3_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block7_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv3[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  add_13 (Add)                   (None, 19, 19, 728)  0           ['block7_sepconv3_bn[0][0]',     
#                                                                   'add_12[0][0]']                 
                                                                                                  
#  block8_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_13[0][0]']                 
#  n)                                                                                               
                                                                                                  
#  block8_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block8_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block8_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block8_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block8_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block8_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block8_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block8_sepconv2_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block8_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv3_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block8_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv3[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  add_14 (Add)                   (None, 19, 19, 728)  0           ['block8_sepconv3_bn[0][0]',     
#                                                                   'add_13[0][0]']                 
                                                                                                  
#  block9_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_14[0][0]']                 
#  n)                                                                                               
                                                                                                  
#  block9_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv1_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block9_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv1[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block9_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block9_sepconv1_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block9_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv2_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block9_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv2[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  block9_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block9_sepconv2_bn[0][0]']     
#  n)                                                                                               
                                                                                                  
#  block9_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv3_act[0][0]']    
#  2D)                                                                                              
                                                                                                  
#  block9_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv3[0][0]']        
#  lization)                                                                                        
                                                                                                  
#  add_15 (Add)                   (None, 19, 19, 728)  0           ['block9_sepconv3_bn[0][0]',     
#                                                                   'add_14[0][0]']                 
                                                                                                  
#  block10_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_15[0][0]']                 
#  on)                                                                                              
                                                                                                  
#  block10_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv1_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block10_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv1[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block10_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block10_sepconv1_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block10_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv2_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block10_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv2[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block10_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block10_sepconv2_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block10_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv3_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block10_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv3[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  add_16 (Add)                   (None, 19, 19, 728)  0           ['block10_sepconv3_bn[0][0]',    
#                                                                   'add_15[0][0]']                 
                                                                                                  
#  block11_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_16[0][0]']                 
#  on)                                                                                              
                                                                                                  
#  block11_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv1_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block11_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv1[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block11_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block11_sepconv1_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block11_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv2_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block11_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv2[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block11_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block11_sepconv2_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block11_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv3_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block11_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv3[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  add_17 (Add)                   (None, 19, 19, 728)  0           ['block11_sepconv3_bn[0][0]',    
#                                                                   'add_16[0][0]']                 
                                                                                                  
#  block12_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_17[0][0]']                 
#  on)                                                                                              
                                                                                                  
#  block12_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv1_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block12_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv1[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block12_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block12_sepconv1_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block12_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv2_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block12_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv2[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block12_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block12_sepconv2_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block12_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv3_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block12_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv3[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  add_18 (Add)                   (None, 19, 19, 728)  0           ['block12_sepconv3_bn[0][0]',    
#                                                                   'add_17[0][0]']                 
                                                                                                  
#  block13_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_18[0][0]']                 
#  on)                                                                                              
                                                                                                  
#  block13_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block13_sepconv1_act[0][0]']   
#  v2D)                                                                                             
                                                                                                  
#  block13_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block13_sepconv1[0][0]']       
#  alization)                                                                                       
                                                                                                  
#  block13_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block13_sepconv1_bn[0][0]']    
#  on)                                                                                              
                                                                                                  
#  block13_sepconv2 (SeparableCon  (None, 19, 19, 1024  752024     ['block13_sepconv2_act[0][0]']   
#  v2D)                           )                                                                 
                                                                                                  
#  block13_sepconv2_bn (BatchNorm  (None, 19, 19, 1024  4096       ['block13_sepconv2[0][0]']       
#  alization)                     )                                                                 
                                                                                                  
#  conv2d_300 (Conv2D)            (None, 10, 10, 1024  745472      ['add_18[0][0]']                 
#                                 )                                                                 
                                                                                                  
#  block13_pool (MaxPooling2D)    (None, 10, 10, 1024  0           ['block13_sepconv2_bn[0][0]']    
#                                 )                                                                 
                                                                                                  
#  batch_normalization_300 (Batch  (None, 10, 10, 1024  4096       ['conv2d_300[0][0]']             
#  Normalization)                 )                                                                 
                                                                                                  
#  add_19 (Add)                   (None, 10, 10, 1024  0           ['block13_pool[0][0]',           
#                                 )                                 'batch_normalization_300[0][0]']
                                                                                                  
#  block14_sepconv1 (SeparableCon  (None, 10, 10, 1536  1582080    ['add_19[0][0]']                 
#  v2D)                           )                                                                 
                                                                                                  
#  block14_sepconv1_bn (BatchNorm  (None, 10, 10, 1536  6144       ['block14_sepconv1[0][0]']       
#  alization)                     )                                                                 
                                                                                                  
#  block14_sepconv1_act (Activati  (None, 10, 10, 1536  0          ['block14_sepconv1_bn[0][0]']    
#  on)                            )                                                                 
                                                                                                  
#  block14_sepconv2 (SeparableCon  (None, 10, 10, 2048  3159552    ['block14_sepconv1_act[0][0]']   
#  v2D)                           )                                                                 
                                                                                                  
#  block14_sepconv2_bn (BatchNorm  (None, 10, 10, 2048  8192       ['block14_sepconv2[0][0]']       
#  alization)                     )                                                                 
                                                                                                  
#  block14_sepconv2_act (Activati  (None, 10, 10, 2048  0          ['block14_sepconv2_bn[0][0]']    
#  on)                            )                                                                 
                                                                                                  
#  avg_pool (GlobalAveragePooling  (None, 2048)        0           ['block14_sepconv2_act[0][0]']   
#  2D)                                                                                              
                                                                                                  
#  predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']               
                                                                                                  
# ==================================================================================================
# Total params: 22,910,480
# Trainable params: 22,855,952
# Non-trainable params: 54,528
# __________________________________