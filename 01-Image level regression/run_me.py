#Solve sea lions counting problem as regression problem on whole image
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2
import os

n_classes= 5
batch_size= 12
epochs= 150
image_size= 512
model_name= 'cnn_regression_150epochs_'

def read_ignore_list():
    df_ignore= pd.read_csv('kaggle_data/mismatched_train_images.txt')
    ignore_list= df_ignore['train_id'].tolist()
    
    return ignore_list
    
#Just remove images from mismatched_train_images.txt
def load_data(dir_path):
    df= pd.read_csv('train.csv')
    
    ignore_list= read_ignore_list()
    n_train_images= 948
        
    image_list=[]
    y_list=[]
    for i in range(0,n_train_images):
        if i not in ignore_list:
            image_path= os.path.join(dir_path, str(i)+'.jpg')
            print(image_path)
            img = cv2.imread(image_path)
            print('img.shape',img.shape)
            image_list.append(img)
           
            row= df.ix[i] 
            y_row= np.zeros((5))
            y_row[0]= row['adult_males']
            y_row[1]= row['subadult_males']
            y_row[2]= row['adult_females']
            y_row[3]= row['juveniles']
            y_row[4]= row['pups']
            y_list.append(y_row)
            
    x_train= np.asarray(image_list)
    y_train= np.asarray(y_list)
    
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)

    return x_train,y_train
    
def get_model():
    input_shape = (image_size, image_size, 3)
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Conv2D(n_classes, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    
    print (model.summary())
    #sys.exit(0) #

    model.compile(loss=keras.losses.mean_squared_error,
            optimizer= keras.optimizers.Adadelta())
             
    return model

def train(trainPath, validOn=False):
    model= get_model()
    
    x_train, y_train= load_data(trainPath)
    
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)
    
    if validOn:
        x_train, x_valid, y_train, y_valid = \
                train_test_split(x_train,y_train,  test_size=0.25)
                
        model.fit_generator(
                    datagen.flow(x_train, y_train, 
                                 batch_size=batch_size),
                    validation_data=datagen.flow(x_valid, y_valid, 
                                                 batch_size=batch_size),
                    validation_steps = 1,
                    steps_per_epoch=len(x_train) / batch_size, 
                    epochs=epochs)
    else:
        model.fit_generator(
                    datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, 
                    epochs=epochs)

   
    model.save(model_name+'_model512x512_V2_150ep.h5')
 
def create_submission(testPath, name, train=False):
    model = load_model(model_name+'_model512x512_V2_150ep.h5')
    
    if train:
        n_test_images = 948
    else:
        n_test_images= 18636
    
    pred_arr= np.zeros((n_test_images,n_classes),np.int32)
    for k in range(0,n_test_images):
        image_path= testPath+str(k)+'.jpg'
        print(image_path)
        
        img= cv2.imread(image_path)
        img= img[None,...]
        pred= model.predict(img)
        pred= pred.astype(int)
        
        pred_arr[k,:]= pred
        
    print('pred_arr.shape', pred_arr.shape)
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id']= range(0,n_test_images)
    df_submission['adult_males']= pred_arr[:,0]
    df_submission['subadult_males']= pred_arr[:,1]
    df_submission['adult_females']= pred_arr[:,2]
    df_submission['juveniles']= pred_arr[:,3]
    df_submission['pups']= pred_arr[:,4]
    df_submission.to_csv(name,index=False)
    

trainPath1 = 'C:\\SeaLions\\train_images_512x512\\'
testPath1 = 'C:\\SeaLions\\test_images_512x512\\' 


#%% Train

train(trainPath=trainPath1, validOn=True)


#%% Pred train

create_submission(testPath=trainPath1,
                  name='train_reg_512x512_V2.csv', train=True)


#%% Pred test

create_submission(testPath=testPath1,
                  name='testing_reg_512x512_V2.csv', train=False)