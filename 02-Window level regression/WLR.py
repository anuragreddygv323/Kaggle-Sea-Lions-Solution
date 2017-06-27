# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:58:18 2017

@author: Gareth

- Fitting image level regression model using non-reseized 512x512 windows
- Lions are using extracted from each window
    - Based on https://www.kaggle.com/radustoicescu/get-coordinates-using-blob-detection
- Used to train convolutional neural net in Keras
    -https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/33900#latest-196245

"""

#%% Imports

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.feature
import pandas as pd

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, \
    UpSampling2D, GlobalAveragePooling2D, Input, concatenate, \
    MaxPooling2D, Lambda, Cropping2D
from keras.utils import np_utils

from keras.models import Model
from keras import backend as K
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from collections import Counter


#%% Define class dictionary and paths

paths = {'train': 'I:\\SeaLions\\Train\\',
       'trainD': 'I:\\SeaLions\\TrainDotted\\',
       'test': 'G:\\Test'}
       
classes = {"adult_males" : 0 ,
         "subadult_males" : 1,
         "adult_females" : 2,
         "juveniles" : 3,
         "pups" : 4,
         "error" : 5,
         "none" : 6}      
       
         
#%% Basic functions for loading train images
       
def files(path):
       return os.listdir(path)
     
       
def name2ID(s):
    """
    Convert file name to ID.
    """
    s = s.split('.')
    return int(s[0])

    
def path2ID(s):
    """
    Convert a path to ID.
    """
    return  int(s.split('\\')[-1].split('.jpg')[0])     
    
    
def ID2Fn(path, i):
    """
    Turn an ID in to a file name.
    """
    return path+'\\'+str(i)+'.jpg'

    
       
def load(fn, plotOn=False):
    """
    Load image using cv2
    """
    im = cv2.imread(fn)
    #im = im[:,:,(2,1,0)].astype(np.uint8)
    if plotOn:
        plt.imshow(im)
        plt.show()
    
    return im

    
def maskOut(im, imD, plotOn=False):
    """
    Mask out blackened regions from Train Dotted
    """
    im2 = im.copy()
    mask = np.sum(imD, axis=2)==0
    im2[mask, 0 ] = 0
    im2[mask, 1 ] = 0
    im2[mask, 2 ] = 0

    if plotOn:
        plt.imshow(im2)
        plt.imshow()
    
    return im # im???
    

def maskOut2(tar, maskIm1, maskIm2, plotOn=False):
    """
    Mask out blackened regions from Train Dotted
    """
    mask_1 = cv2.cvtColor(maskIm1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 17] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(maskIm2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 17] = 0
    mask_2[mask_2 > 0] = 255
    
    tar = cv2.bitwise_or(tar, tar, mask=mask_1)
    tar = cv2.bitwise_or(tar, tar, mask=mask_2) 
    
    # Remove jpg artefact
    tar[tar < 5] = 0
    
    if plotOn:
        plt.imshow(mask_1)
        plt.show()
        plt.imshow(mask_2)
        plt.show()
        plt.imshow(tar)
        plt.show()
    
    return tar
    
    
def diff(im, imD, plotOn=False):
    """
    Get difference between train image and dotted
    """
    d = cv2.absdiff(im,imD)
    if plotOn:
        plt.imshow(d)
        plt.show()
    return d

def gs(im, plotOn=False):
    """
    Convert to grayscale
    """
    bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
     
    if plotOn:
        plt.imshow(bw)
        plt.show()
    return bw
     
    
#%% Image process 1 file
"""
Load image and dotted image
Take difference
Mask out any blacked out bits
"""

# Find files
trainFiles = files(paths['train'])    

# Pick a file
p = 0

# Load image
fn = paths['train'] + trainFiles[p]
im = load(fn, True)

# Load dotted
fnD = paths['trainD'] + trainFiles[p]
imD = load(fnD, True)

# Get dots
d = diff(im, imD, True)

# Mask dots
m = maskOut2(d, im, imD, True)

# Convert to bw
bw = gs(m, True)


#%% Functions to extract class counts from training images

def getBlobs(bw):
    """
    Use blob extractor to find blob coords in a black and white image
    """

    blobs = skimage.feature.blob_log(bw, 
                                 min_sigma=3, 
                                 max_sigma=4, 
                                 num_sigma=1, 
                                 threshold=0.02)
    return blobs


def returnClass(blob, im, classes, plotOn=False):
    """
    Return class number and string of a single blob colour from class dict
    Averages colours around dot centre
    """
    # Get blob coords and size
    y, x, s = blob
    # Get individual channels of im
    b, g, r = im[int(y)][int(x)][:]

    # Get average values of each channel a few pixels around centre
    b = np.mean(im[int(y-2):int(y+3), int(x-3):int(x+2), 0], dtype=np.int32)
    g = np.mean(im[int(y-2):int(y+3), int(x-3):int(x+2), 1], dtype=np.int32)
    r = np.mean(im[int(y-2):int(y+3), int(x-3):int(x+2), 2], dtype=np.int32)
    
    if plotOn:
        plt.imshow(im[int(y-2):int(y+2), int(x-2):int(x+2), (2,1,0)])
        plt.show()
        plt.imshow(im[int(y-50):int(y+50), int(x-50):int(x+50), (2,1,0)])
        plt.show()
        
    # Get class depending on colour    
    if r > 200 and g < 50 and b < 50: # RED; adult_males 
        s = 'adult_males'    
        n = classes[s]
        col = 'RED'
        
    elif r > 200 and g < 50 and b > 200: # MAGENTA; subadult_males 
        s = 'subadult_males'    
        n = classes[s]
        col = 'MAGENTA'
        
    elif 50 < r < 150 and g < 100 and b < 50: # BROWN; adult_females 
        s = 'adult_females'  
        n = classes[s]
        col = 'BROWN'
        
    elif r < 100 and g < 100 and 125 < b < 225: # BLUE; juveniles 
        s = 'juveniles'    
        n = classes[s]
        col = 'BLUE'

    elif r < 100 and 100 < g and b < 100: # GREEN; pups
        s = 'pups'    
        n = classes[s]
        col = 'GREEN'

    else:
        s = 'none'
        n = classes[s]
        col = 'UNKNOWN'
        
    if plotOn:
        print('Above is:', col, s, n)
        
    return(n, s)


def returnClasses(blobs, im, classes, plotOn=False):
    """
    For multiple blobs, return and count classes
    """
    
    listName = []
    listNum = []
    # For each blob
    for blob in blobs:
        # Get class
       cla1, cla2 = returnClass(blob, im, classes, plotOn)
       # Append to list
       listName.append(cla2)
       listNum.append(cla1)
    
    # Add to found lions to dict       
    counts = {} 
    for c in classes.keys():
        counts[c] = listName.count(c)
       
    return counts, listNum, listName

    
def testCounts(counts, GT):
    """
    Compare counts from an image to ground truth.
    """
    for c in classes.keys():
        print(c, ':', counts[c], '|', int(GT.get(str(c), default=969696)))
        
    
#%% Count for previously loaded single file

# Extract blobs from whole image
blobs = getBlobs(bw)

# Get count of classes
counts, lNum, lName = returnClasses(blobs, imD, classes, plotOn=False)

# Load train.csv
GT = pd.read_csv('train.csv')

# Get index of file in train.csv
GTIdx = path2ID(fn) ==  GT.train_id

# Comapre counted lions to GT
testCounts(counts, GT.loc[GTIdx,:])


#%% Test so far - run all for one file

# Pick a file
p = 40 

# Load
GT = pd.read_csv('train.csv')
trainFiles = files(paths['train']) 

fn = paths['train'] + trainFiles[p]
im = load(fn, True)

fnD = paths['trainD'] + trainFiles[p]
imD = load(fnD, True)

# Process
d = diff(im, imD, False)
m = maskOut2(d, im, imD, False)
bw = gs(m, False)
blobs = getBlobs(bw)

# Count
counts, lNum, lName = returnClasses(blobs, imD, classes, False)

# Compare
# GTIdx = int(fnD.split('\\')[-1].split('.jpg')[0]) == GT.train_id
GTIdx = path2ID(fn) ==  GT.train_id
testCounts(counts, GT.loc[GTIdx,:])


#%% Functions to count training set

def splitCount(fn, fnD, size=[512,512], plotOn=False):

    """
    Sliding window of size size, plus labels as counted
    One image as input
    """
    
    # Load and process to bw step
    im = load(fn, False)
    imD = load(fnD, False)
    d = diff(im, imD, False)
    m = maskOut2(d, im, imD, False)
    bw = gs(m, False)

    # Calc steps required
    x = im.shape[1]
    y = im.shape[0]
    stepsX = np.floor(x/size[1])
    stepsY = np.floor(y/size[0])

    # Preallocate output arrays
    out = np.empty(shape=(int(stepsX*stepsY),size[0],size[1],3), 
                          dtype=np.uint16)
    countsSub = np.empty(shape=(int(stepsX*stepsY), 7), 
                          dtype=np.uint32)
    # To mark bad/empty
    marked = np.zeros(shape=(int(stepsX*stepsY)), dtype=np.bool)
    # Counter to make adding dicts easier
    imCounts = Counter()
    
    linIdx = -1
    for xi in range(0, int(stepsX)):
        for yi in range(0, int(stepsY)):
            
            if plotOn:
                print('-'*30)
                
            linIdx += 1
            
            # Keep window from image for training
            out[linIdx,:,:,:] = im[int(yi*size[0]):int((yi+1)*size[0]),
                                   int(xi*size[1]):int((xi+1)*size[1]), 
                                   :]
            
            # Get dotted version, to get class, not keeping
            imDSub = imD[int(yi*size[0]):int((yi+1)*size[0]),
                                   int(xi*size[1]):int((xi+1)*size[1]), 
                                   :]
                                   
            # Get corresponding bw for blob counting                                   
            bwSub = bw[int(yi*size[0]):int((yi+1)*size[0]),
                                   int(xi*size[1]):int((xi+1)*size[1])]
            
            # If all black, mark empty                                   
            if np.sum(bwSub) == 0:
                marked[linIdx] = True
                countsSub[linIdx, :] = np.zeros(shape=(1,7))
            else:
                # Data here, count
                # Get blobks
                blobs = getBlobs(bwSub)
                # Count lions
                cts, lNum, lName = returnClasses(blobs, imDSub, 
                                                 classes, plotOn)    
                # Added counted lions from window to total count for image
                imCounts = imCounts+Counter(cts)
                
                # Care: Order
                # Also save in table for idiot checking
                countsSub[linIdx, :] = np.array(list(cts.values()))
                      
            if plotOn:                       
                plt.imshow(out[linIdx,:,:,:])    
                plt.show()                   
                print(linIdx)
                print(int(yi*size[1]), int((yi+1)*size[0]))
                print(int(xi*size[0]), int((xi+1)*size[1]))
    
    
    return out, imCounts, countsSub, marked
    
    
#%% Test splitCount on single image

# Pick file
p = 3

# Set paths
trainFiles = files(paths['train']) 
fn = paths['train'] + trainFiles[p]
fnD = paths['trainD'] + trainFiles[p]
GT = pd.read_csv('train.csv')
 
# Count 
imTrain, imCounts, countsSub, imMarked = splitCount(fn, fnD, plotOn=False)

# Compare sum of counts to GT
GTIdx = path2ID(fn) ==  GT.train_id
testCounts(imCounts, GT.loc[GTIdx,:])


#%% Compile train
# Compile training set by collecting all 512x512 windows from each image, 
# counting the lions in the windows along the way
# Drop a random proportion of windows with no lions

GT = pd.read_csv('train.csv')

# Ignore these images 
bad = [paths['train']  + '3.jpg',
paths['train']  + '7.jpg',
paths['train']  + '9.jpg',
paths['train']  + '21.jpg',
paths['train']  + '30.jpg',
paths['train']  + '34.jpg',
paths['train']  + '71.jpg',
paths['train']  + '81.jpg',
paths['train']  + '89.jpg',
paths['train']  + '97.jpg',
paths['train']  + '151.jpg',
paths['train']  + '184.jpg',
paths['train']  + '215.jpg',
paths['train']  + '234.jpg',
paths['train']  + '242.jpg',
paths['train']  + '268.jpg',
paths['train']  + '290.jpg',
paths['train']  + '311.jpg',
paths['train']  + '331.jpg',
paths['train']  + '344.jpg',
paths['train']  + '380.jpg',
paths['train']  + '384.jpg',
paths['train']  + '406.jpg',
paths['train']  + '421.jpg',
paths['train']  + '469.jpg',
paths['train']  + '475.jpg',
paths['train']  + '490.jpg',
paths['train']  + '499.jpg',
paths['train']  + '507.jpg',
paths['train']  + '530.jpg',
paths['train']  + '531.jpg',
paths['train']  + '605.jpg',
paths['train']  + '607.jpg',
paths['train']  + '614.jpg',
paths['train']  + '621.jpg',
paths['train']  + '638.jpg',
paths['train']  + '644.jpg',
paths['train']  + '687.jpg',
paths['train']  + '712.jpg',
paths['train']  + '721.jpg',
paths['train']  + '767.jpg',
paths['train']  + '779.jpg',
paths['train']  + '781.jpg',
paths['train']  + '794.jpg',
paths['train']  + '800.jpg',
paths['train']  + '811.jpg',
paths['train']  + '839.jpg',
paths['train']  + '840.jpg',
paths['train']  + '869.jpg',
paths['train']  + '882.jpg',
paths['train']  + '901.jpg',
paths['train']  + '903.jpg',
paths['train']  + '905.jpg',
paths['train']  + '909.jpg',
paths['train']  + '913.jpg',
paths['train']  + '927.jpg',
paths['train']  + '946.jpg']

trainFiles = files(paths['train']) 
nFiles = len(trainFiles)

# Set parameters
size = [512,512]


# Use alternative definition of classes here
# Error is not used
classes2 = ["adult_males", "subadult_males", "adult_females", "juveniles", 
            "pups" , "none", "error"]      
nClasses = len(classes2)

# Preallocate an output array with the expectation of 50 tiles per image
# (Total is ~50,000 including windows with no lions)
nPreAl = nFiles*50
allSubImages = np.empty(shape=(nPreAl, size[0], size[1], 3), dtype=np.uint8)
allSubCounts = pd.DataFrame(np.zeros(shape=(nPreAl, nClasses), dtype=np.uint8))
allSubCounts.columns = classes2

sRow = 0
added = 0
totalAdded = 0
for p in range(0, nFiles):
    fn = paths['train'] + trainFiles[p]
    fnD = paths['trainD'] + trainFiles[p]
    
    if fn in bad:
        print('Skipping:', fn+', bad')
        continue
    
    print(fn)
    print(fnD)
    
    # Get counts from windows
    imTrain, imCounts, countsSub, imMarked = splitCount(fn, fnD, plotOn=False)
    GTIdx = path2ID(fn) ==  GT.train_id
    testCounts(imCounts, GT.loc[GTIdx,:])

    # Drop empty windows (all black windows)
    # imMarked contains masked out images
    # np.sum(countsSub, axis=1)==0 contains no-lion images
    keepIdx1 = imMarked==False

    # Drop most empty windows (no lions)
    # # Ignore "none"
    keepIdx2 = (np.sum(countsSub[:,0:6], axis=1)>0) | (np.random.rand(imTrain.shape[0])>0.75)

    # Apply drops, count
    imAdd = imTrain[keepIdx1&keepIdx2,:,:,:]

    countsAdd = countsSub[keepIdx1&keepIdx2,:]
    nAdd = imAdd.shape[0]    
    
    eRow = sRow + nAdd
    if eRow>nPreAl:
        # More lions found than preallocated space
        # Break without adding
        break

    # Add found lions to output arrays  
    allSubImages[sRow:eRow,:,:,:] = imAdd
    for c in allSubCounts:
        if c in list(imCounts.keys()):
            caIdx = list(imCounts.keys()).index(c)
            allSubCounts.ix[sRow:(eRow-1),c] = countsAdd[:,caIdx]

    sRow = eRow+1
    totalAdded += nAdd

    print('Added so far:', totalAdded)
    print('-'*30)
    
    if sRow>nPreAl:
        break
    
    
# If not enough added (ie fewer than preallocated space)
allSubImages = allSubImages[0:totalAdded,:,:,:]
allSubCounts = allSubCounts.loc[0:(totalAdded-1),:]

# Save to disk                                
np.savez('I:\\SeaLions\\trainSetCombined2.npz', 
         allSubImages=allSubImages)
allSubCounts.to_csv('I:\\SeaLions\\trainSetCombined2_SubCounts.csv')


#%% Load
#f = np.load('I:\\SeaLions\\trainSetCombined2.npz')
#allSubImages= f['allSubImages']
#allSubCounts= pd.read_csv('I:\\SeaLions\\trainSetCombined2_SubCounts.csv')

#%% Reduce memory

import gc
gc.collect()


#%% Keras model
# Modified from https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/33900#latest-196245

n_classes = 5
batch_size = 10
epochs = 80
image_size = 512
model_name = 'windows_cnn_regression_updatedClasses_512x512'
    
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

def trainCNN(x_train, y_train, dg=True):
    model= get_model()
    
    if dg:
        datagen = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)
        
        history = model.fit_generator(datagen.flow(x_train, y_train, 
                                                   batch_size=batch_size),
                steps_per_epoch=len(x_train) / batch_size, epochs=epochs)
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size, 
                            epochs=epochs, validation_split=0.2)

    model.save(model_name+'_model2.h5')
    
    return history

    
def create_submission(testPath, name):
    model = load_model(model_name+'_model2.h5')
    
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
  

#%% Train model
# Drop a further proprtion of data and empty data, depending on availale memory

# Shrink training set
nTrain = allSubImages.shape[0]

# Drop some more empties?
keepIdx1 = (np.sum(allSubCounts, axis=1)>0) | (np.random.rand(nTrain)>0.1)

# Drop from remaining set?
keepIdx2 = np.random.rand(nTrain)>0.45
# x_train = allSubImages[keepIdx,:,:,:].astype(np.uint8)
    
print(np.sum(keepIdx1))
print(np.sum(keepIdx2))
print(np.sum(keepIdx1 & keepIdx2))

# Remove unwated classes and order as above in sub prep
# Here dropping none and error
classesUse = classes2[0:-2] 
yTrain = allSubCounts.loc[:, classesUse]

# Train 
# Model is saved in trainCNN
history = trainCNN(allSubImages[keepIdx1&keepIdx2,:,:,:], 
                   yTrain.loc[keepIdx1&keepIdx2,:].values,
                     dg=False)


#%% Load model

model_name = 'windows_cnn_regression_updatedClasses_512x512_model2.h5'
mod = load_model(model_name)


#%% Prediction functions

def splitPred(mod, im, classes, size=[512,512], plotOn=False):
    
    """
    Sliding window of size size, plus labels as counted
    One image as input
    """
    
    # Calc steps required
    x = im.shape[1]
    y = im.shape[0]
    stepsX = np.floor(x/size[1])
    stepsY = np.floor(y/size[0])

    # Preallocate
    predsSub = pd.DataFrame(np.zeros(shape=(int(stepsX*stepsY), 5), 
                          dtype=np.float32))
    
    linIdx = -1
    for xi in range(0, int(stepsX)):
        for yi in range(0, int(stepsY)):
            
            if plotOn:
                print('-'*30)
                
            linIdx += 1
            
            # Keep actual image
            imSub = im[int(yi*size[0]):int((yi+1)*size[0]),
                                   int(xi*size[1]):int((xi+1)*size[1]), 
                                   :]
            
            # If all black, mark empty                                   
            # if np.sum(out[linIdx,:,:,:]) == 0:
            if np.sum(im) == 0:
                predsSub.loc[linIdx, :] = np.zeros(shape=(1,5))
            else:
                # Data here, pred
                preds = mod.predict(np.expand_dims(imSub, axis=0))
                
                # Care: Order
                predsSub.loc[linIdx, :] = preds
                      
            if plotOn:                       
                plt.imshow(imSub[:,:,:])    
                plt.show()                   
                print(linIdx)
                print(int(yi*size[1]), int((yi+1)*size[0]))
                print(int(xi*size[0]), int((xi+1)*size[1]))
                print(classes.keys())
                print(preds)
    
                
    # Sum preds across sub images        
    predsIm = np.sum(predsSub, axis=0)
    
    return predsSub, predsIm.astype(np.uint8)

def testPreds(preds, classes2, GT):
    print(preds)
    
    for c in classes2:
        
        if c in ['error', 'none']:
            n = 9696968
        else:
            pIdx = classes2.index(c)
            n = preds[pIdx]
        
        
        print(c, ':', n, '|', int(GT.get(str(c), default=969696)))    

        
#%% Predict for 1 train

# Pick a file
p = 1

# Prep
GT = pd.read_csv('train.csv')
trainFiles = files(paths['train'])   

# Load single image
fn = paths['train'] + trainFiles[p]
im = load(fn, True)

# Predict
predsSub, predsIm = splitPred(mod, im, classes, plotOn=True)

# Check
GTIdx = path2ID(fn) ==  GT.train_id
testPreds(predsIm, classes2, GT.loc[GTIdx,:])


#%% Predict all train
# Predict for all of the training set

GT = pd.read_csv('train.csv')

trainFiles = files(paths['train']) 
nFiles = len(trainFiles)

size = [512,512]
added = 1


allPreds =  pd.DataFrame(np.zeros(shape=(nFiles, 6), dtype=np.uint8))
allPreds.columns = GT.columns


sRow = 0
totalAdded = 0
cl = classes2
for p in range(0, nFiles):
    print(str(p+1), '/',  str(nFiles))
    print(GT.train_id[p])
    
    # Load
    fn = ID2Fn(paths['train'], GT.train_id[p])
    print(fn)
    im = load(fn, True)

    # Predict
    predsSub, predsIm = splitPred(mod, im, classes, plotOn=False)
    
    # Check
    GTIdx = path2ID(fn) ==  GT.train_id
    testPreds(predsIm, classes2, GT.loc[GTIdx,:])

    # Save
    allPreds.loc[p, 'train_id'] = p
    allPreds.loc[p, 'adult_males'] = predsIm[cl.index('adult_males')]
    allPreds.loc[p, 'subadult_males'] = predsIm[cl.index('subadult_males')]
    allPreds.loc[p, 'adult_females'] = predsIm[cl.index('adult_females')]
    allPreds.loc[p, 'juveniles'] = predsIm[cl.index('juveniles')]
    allPreds.loc[p, 'pups'] = predsIm[cl.index('pups')]

    
    
#%% Predict for test
# Predict for all windows in test set

testFiles = files(paths['test']) 
nFiles = len(testFiles)
GT = pd.read_csv('train.csv')

size = [512, 512]

# Prepare output
allPredsTest = pd.DataFrame(np.zeros(shape=(nFiles, 6), dtype=np.uint8))
allPredsTest.columns = GT.columns

cl = classes2

for p in range(0, nFiles):
    print(str(p+1), '/',  str(nFiles))
    
    # Load
    fn = ID2Fn(paths['test'], p)
    print(fn)
    im = load(fn, False)

    # Predict
    predsSub, predsIm = splitPred(mod, im, classes, plotOn=False)
    
    # Save
    allPredsTest.loc[p, 'train_id'] = p
    allPredsTest.loc[p, 'adult_males'] = predsIm[cl.index('adult_males')]
    allPredsTest.loc[p, 'subadult_males'] = predsIm[cl.index('subadult_males')]
    allPredsTest.loc[p, 'adult_females'] = predsIm[cl.index('adult_females')]
    allPredsTest.loc[p, 'juveniles'] = predsIm[cl.index('juveniles')]
    allPredsTest.loc[p, 'pups'] = predsIm[cl.index('pups')]


"""
Use code here to save/load if prediction is stopped before completion

#%% SAVE SO FAR
p = 11722 
file = 'allPredsTest_done.csv'
allPredsTest.to_csv(file, index=false)


#%% LOAD SO FAR

p = 11722
classes2=["adult_males", "subadult_males", "adult_females", "juveniles", "pups" , "none", "error"]
GT = pd.read_csv('train.csv')
mod = load_model('windows_cnn_regression_updatedClasses_512x512_model2.h5')
allPredsTest = pd.read_csv('allPredsTest_' + str(p) + ' .csv')
allPredsTest = allPredsTest.drop('Unnamed: 0', axis=1)
"""

#%% Create submission

sub = allPredsTest.iloc[:,1::]
sub.columns = ['test_id'] + list(sub.columns[1::])
sub.to_csv('sub.csv', index=False)

