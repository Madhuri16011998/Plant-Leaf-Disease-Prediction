import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Tomato_Leaf_Disease_Prediction/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

tomato_plant = cv2.imread('C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Tomato_Leaf_Disease_Prediction/Dataset/test/Tomato___Bacterial_spot (1).JPG')
test_image = cv2.resize(tomato_plant, (128,128)) # load image 
  
test_image = img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased palnt or not
  
pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Tomato - Bacteria Spot Disease")
       
elif pred==1:
    print("Tomato - Early Blight Disease")
        
elif pred==2:
    print("Tomato - Healthy and Fresh")
        
elif pred==3:
    print("Tomato - Late Blight Disease")
       
elif pred==4:
    print("Tomato - Leaf Mold Disease")
        
elif pred==5:
    print("Tomato - Septoria Leaf Spot Disease")
        
elif pred==6:
    print("Tomato - Target Spot Disease")
        
elif pred==7:
      print("Tomato - Tomoato Yellow Leaf Curl Virus Disease")
elif pred==8:
      print("Tomato - Tomato Mosaic Virus Disease")
        
elif pred==9:
      print("Tomato - Two Spotted Spider Mite Disease")
