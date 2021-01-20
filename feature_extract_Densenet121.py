
# feature extraction from Densnet121 4-class model
# 1) finetuned 
# 2) pre-trained 

from keras.applications.densenet import DenseNet121, preprocess_input
import keras

import time
import cv2
import keras.backend as K
import numpy as np
import glob
import os
import csv

from keras import optimizers
from scipy import interp
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image

load_model_from_pretrained= True

model_weights_path = "./models/DenseNet121/model_.37-0.81_4class_DenseNet121.hdf5"
model_name='DenseNet121'     
features_CSV_path= "./models/" + model_name 
image_size = (224, 224)

mode='train' # train or test 
train_path    = "./data_features/" + mode

num_classes=4

# variables to hold features and labels
features = []
labels   = []

def scanfolder(myPath, Ext):
    filesList = list();
    for path, dirs, files in os.walk(myPath):
        for f in files:
            if f.endswith(Ext):
                #print(os.path.join(path, f))
                filesList.append(os.path.join(path, f))
    return filesList

Ext = ('png','jpg','JPEG','bmp','JPG','tif') 

##-------Load Pre-train Densenet121 model bised on imagenet weights ------------###
def load_model_pre():
    base_model = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)
    x = keras.layers.AveragePooling2D((7,7))(base_model.output)
    x_newfc = keras.layers.Flatten()(x)
    model_pre = keras.models.Model(input=base_model.input, output=x_newfc)
    
    return model_pre


##-------Load Finetuned Densenet121 model bised on new weights ------------###

def load_model():
    base_model = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)
    x = keras.layers.AveragePooling2D((7,7))(base_model.output)  
    x_newfc = keras.layers.Flatten()(x)
    x_newfc = keras.layers.Dense(num_classes, activation='softmax', name='fc_new')(x_newfc)
    model1 = keras.models.Model(input=base_model.input, output=x_newfc)
    model = keras.models.Model(input=base_model.input, output=model1.get_layer('flatten_1').output)
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(model_weights_path, by_name=True)
    model.summary()
    return model

##-----------loading Model----------------------------------##

if load_model_from_pretrained:
    model = load_model_pre()
else: 
    model = load_model()
#-------------------------------------------------------------


# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])
class_label=[]  
    
print('analysing images...')
count = 1
for i, label in enumerate(train_labels):   ###############
  cur_path = train_path + "/" + label
  filesList = scanfolder(cur_path, Ext)
  #count = 1
  for image_path in filesList:

      try:
        img = image.load_img(image_path, target_size=image_size)       
        x = image.img_to_array(img)       
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feature = model.predict(x)
        class_label=[]  
        
        features.append(feature)
        
        if label=='E':
            class_label=0
            
        if label=='NL': 
            class_label=1

        if label=='PE':
            class_label=2
            
        if label=='PR': 
            class_label=3

        if (count % 1000 ==0 and count > 1):
            print("processed image : " +  str(count))
            

        labels.append(class_label)        
        count += 1
        

    except:
        print("Unexpected error") 
        print(image_path)

for i in range(0, len(features)):
    fea = np.array(features[i])
    
    la  = np.array([labels[i]])
    
    nfea = np.concatenate((la, fea), axis=None)
    features[i] = nfea

print('save extracted features in .csv format')

if load_model_from_pretrained:
    np.savetxt((features_CSV_path + '/pre_'  + model_name +  '_' + mode +   '.csv'), np.array(features), delimiter=',',fmt='%1.4f')

else : 
    np.savetxt((features_CSV_path + '/'  + model_name +  '_' + mode +   '.csv'), np.array(features), delimiter=',',fmt='%1.4f')

   
    






       
    