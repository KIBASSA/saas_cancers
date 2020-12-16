
import glob
from random import shuffle
import fnmatch
from cv2 import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from keras.utils.np_utils import to_categorical

class DataLoader(object):

    def proc_images(self, imagePatches,classZero,classOne, lowerIndex, upperIndex):
        """
        Returns two arrays: 
            x is an array of resized images
            y is an array of labels
        """ 
        x = []
        y = []
        WIDTH = 50
        HEIGHT = 50
        for index, img in enumerate(imagePatches[lowerIndex:upperIndex]):
            if index % 1000 == 0:
                print("index : ", index)
            full_size_image = cv2.imread(img)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            if img in classZero:
                y.append(0)
            elif img in classOne:
                y.append(1)
            else:
                return
        return x,y 

    def load(self, path, limit=300000):

        imagePatches = glob.glob(path, recursive=True)
        shuffle(imagePatches)

        patternZero = '*class0.png'
        patternOne = '*class1.png'
        classZero = fnmatch.filter(imagePatches, patternZero)
        classOne = fnmatch.filter(imagePatches, patternOne)

        X,Y = self.proc_images(imagePatches, classZero, classOne, 0, 160000)
        df = pd.DataFrame()
        df["images"]=X
        df["labels"]=Y
        X2=df["images"]
        Y2=df["labels"]
        X2=np.array(X2)
        #imgs0=[]
        #imgs1=[]
        #imgs0 = X2[Y2==0] # (0 = no IDC, 1 = IDC)
        #imgs1 = X2[Y2==1]

        dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

        X=np.array(X)
        X=X/255.0

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X_train = X_train[0:limit] 
        Y_train = Y_train[0:limit]
        X_test = X_test[0:limit] 
        Y_test = Y_test[0:limit]

        # Deal with imbalanced class sizes below
        # Make Data 1D for compatability upsampling methods
        X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
        X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
        X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
        X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)


        ros = RandomUnderSampler(sampling_strategy='auto')
        X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
        X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

        # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
        Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 2)
        Y_testRosHot = to_categorical(Y_testRos, num_classes = 2)

        print("len(X_trainRos) : ", len(X_trainRos))
        for i in range(len(X_trainRos)):
            height, width, channels = 50, 50, 3
            X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos), height, width, channels)


        for i in range(len(X_testRos)):
            height, width, channels = 50, 50, 3
            X_testRosReshaped = X_testRos.reshape(len(X_testRos), height, width, channels)
        
        return X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot