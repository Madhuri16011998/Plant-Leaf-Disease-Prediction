from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import os

#basic cnn
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Tomato_Leaf_Disease_Prediction/Dataset/train', # relative path from working directoy
                                                 target_size = (128, 128),
                                                 batch_size = 6, class_mode = 'categorical')
valid_set = test_datagen.flow_from_directory('C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Tomato_Leaf_Disease_Prediction/Dataset/val', # relative path from working directoy
                                             target_size = (128, 128), 
                                        batch_size = 3, class_mode = 'categorical')

labels = (training_set.class_indices)
print(labels)


classifier.fit_generator(training_set,
                         steps_per_epoch = 20,
                         epochs = 50,
                         validation_data=valid_set

                         )

classifier_json=classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
    classifier.save_weights("my_model_weights.h5")
    classifier.save("model.h5")
    print("Saved model to disk")

'''
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

img = cv2.imread('C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Leaf_disease/data/d (7)_iaip.jpg')
img_resize = cv2.resize(img, (128,128))


CV2 reads an image in BGR format. We need to convert it to RGB
b,g,r = cv2.split(img_resize)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb


plt.imshow(rgb_img)
label_map = (training_set.class_indices)

print(label_map)
img_rank4 = np.expand_dims(rgb_img/255, axis=0)

classifier.predict(img_rank4)
h = list(label_map.keys())[classifier.predict_classes(img_rank4)[0]]
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(img, h, (10, 30), font, 1.0, (0, 0, 255), 1)
cv2.imshow(h,img)

print(h)
'''
