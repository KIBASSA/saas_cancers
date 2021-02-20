
from cv2 import cv2
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from abstract_trainer import AbstractModelTrainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Conv2D,MaxPooling2D,Flatten
import itertools
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras
import glob
from random import shuffle
import fnmatch
from keras.utils.np_utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import sklearn
import os
class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

class PaulMooneyCreditModelTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
        self.BATCH_SIZE_TRAINING = 32
        self.BATCH_SIZE_VALIDATION = 32
        self.BATCH_SIZE_TESTING = 1
    
    def proc_images(self, imagePatches, classZero, classOne, lowerIndex, upperIndex):
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

    def plotKerasLearningCurve(self):
        plt.figure(figsize=(10,5))
        metrics = np.load('logs.npy', allow_pickle = True)[()]
        filt = ['acc'] # try to add 'loss' to see the loss learning curve
        for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
            l = np.array(metrics[k])
            plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
            x = np.argmin(l) if 'loss' in k else np.argmax(l)
            y = l[x]
            plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
            plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
        plt.legend(loc=4)
        plt.axis([0, None, None, None])
        plt.grid()
        plt.xlabel('Number of epochs')

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): # pylint: disable=no-member
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure(figsize = (5,5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_learning_curve(self, history):
        plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./accuracy_curve.png')
        #plt.clf()
        # summarize history for loss
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./loss_curve.png')

    def get_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape,strides=2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        return model

    def train(self,input_data, model_candidate_folder):
        
        pattern_images = os.path.join(input_data,'train/**/*.png')
        print("pattern_images :", pattern_images)
        imagePatches = glob.glob(pattern_images, recursive=True)
        shuffle(imagePatches)
        print("imagePatches :", len(imagePatches))
        """ Process Data
        """

        patternZero = '*class0.png'
        patternOne = '*class1.png'
        classZero = fnmatch.filter(imagePatches, patternZero)
        classOne = fnmatch.filter(imagePatches, patternOne)
        print("classZero :", len(classZero))
        print("classOne :", len(classOne))

        print("IDC(-)\n\n",classZero[0:5],'\n')
        print("IDC(+)\n\n",classOne[0:5])

        #90000
        X,Y = self.proc_images(imagePatches, classZero, classOne, 0, 160000)
        df = pd.DataFrame()
        df["images"]=X
        df["labels"]=Y
        X2=df["images"]
        Y2=df["labels"]
        X2=np.array(X2)
        imgs0=[]
        imgs1=[]
        imgs0 = X2[Y2==0] # (0 = no IDC, 1 = IDC)
        imgs1 = X2[Y2==1] 

        dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

        X=np.array(X)
        X=X/255.0

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Reduce Sample Size for DeBugging
        print("len(X_train) :", len(X_train))
        X_train = X_train[0:300000] 
        Y_train = Y_train[0:300000]
        X_test = X_test[0:300000] 
        Y_test = Y_test[0:300000]
        print("len(X_train) :", len(X_train))

        # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
        Y_trainHot = to_categorical(Y_train, num_classes = 2)
        Y_testHot = to_categorical(Y_test, num_classes = 2)

        # Deal with imbalanced class sizes below
        # Make Data 1D for compatability upsampling methods
        X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
        X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
        X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
        X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)

        print("np.unique(Y_train) : ", np.unique(Y_train))
        #ros = RandomOverSampler(ratio='auto')
        print("len(X_trainFlat) : ", len(X_trainFlat))
        ros = RandomUnderSampler(sampling_strategy='auto')
        X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
        X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

        # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
        Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 2)
        Y_testRosHot = to_categorical(Y_testRos, num_classes = 2)

        print("len(X_trainRos) : ", len(X_trainRos))
        for i in range(len(X_trainRos)):
            height, width, channels = 50,50,3
            X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
        
        print("X_trainRos Shape: ",X_trainRos.shape)
        print("X_trainRosReshaped Shape: ",X_trainRosReshaped.shape)

        for i in range(len(X_testRos)):
            height, width, channels = 50,50,3
            X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
        
        #Step 4: Define Helper Functions for the Classification Task

        
        class_weight1 = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train) # pylint: disable=missing-kwoa,too-many-function-args
        print("Old Class Weights: ",class_weight1)
        class_weight2 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos) # pylint: disable=missing-kwoa,too-many-function-args
        #print("New Class Weights: ",class_weight2)


        batch_size = 128
        num_classes = 2
        epochs = 8
        #img_rows, img_cols = a.shape[1],a.shape[2]
        img_rows,img_cols=50,50
        input_shape = (img_rows, img_cols, 3)
        model = self.get_model(input_shape, num_classes)
        
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        cb_checkpointer = keras.callbacks.ModelCheckpoint(filepath = classifier_file, monitor = 'val_accuracy', save_best_only = True, mode = 'auto')

        history = model.fit_generator(datagen.flow(X_trainRosReshaped,Y_trainRosHot, batch_size=32),
                            steps_per_epoch=len(X_trainRosReshaped) / 32, 
                            epochs=epochs,
                            class_weight=class_weight2, 
                            validation_data = [X_testRosReshaped, Y_testRosHot],
                            callbacks = [MetricsCheckpoint('logs'), cb_checkpointer])

        score = model.evaluate(X_testRosReshaped,Y_testRosHot, verbose=0)
        print('\nKeras CNN #1C - accuracy:', score[1],'\n')
        y_pred = model.predict(X_testRosReshaped)
        map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
        print('\n', sklearn.metrics.classification_report(np.where(Y_testRosHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    
        Y_pred_classes = np.argmax(y_pred,axis=1)
        Y_true = np.argmax(Y_testRosHot,axis=1) 
        self.plotKerasLearningCurve()
        plt.show()  
        self.plot_learning_curve(history)
        plt.show()
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
        self.plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 
        plt.show()

#runKerasCNNAugment(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,2,class_weight2)
