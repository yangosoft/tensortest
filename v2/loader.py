
# Create a new model instance
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


import tensorflowjs as tfjs



size=94
new_model = tf.keras.models.load_model('model/model1')

# Check its architecture
new_model.summary()




import numpy as np

 
import keras.utils as image

ImagePath='./images/right_look/right_(8).jpg'
test_image=image.load_img(ImagePath,target_size=(size,size))
test_image=image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=new_model.predict(test_image,verbose=0)
#print(training_set.class_indices)
 
print('####'*10)
print('Prediction is: ',str(result))

