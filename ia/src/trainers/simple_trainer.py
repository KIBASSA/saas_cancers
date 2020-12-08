from generator import generator_network
from discriminator import disc_network
from global_helpers import AzureMLLogsProvider
from abstract_trainer import AbstractModelTrainer
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D,MaxPooling2D,Flatten
import os
import numpy as np
from tqdm import tqdm
from enum import Enum
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from collections import Counter
import glob
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import itertools
import sklearn
from sklearn.utils import class_weight
import fnmatch
import cv2
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from random import shuffle
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
    
    def proc_images(self, imagePatches,classZero,classOne, lowerIndex,upperIndex):
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
                          cmap=plt.cm.Blues):
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

        
        class_weight1 = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
        #print("Old Class Weights: ",class_weight1)
        class_weight2 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
        #print("New Class Weights: ",class_weight2)


        batch_size = 128
        num_classes = 2
        epochs = 8
    #   img_rows, img_cols = a.shape[1],a.shape[2]
        img_rows,img_cols=50,50
        input_shape = (img_rows, img_cols, 3)
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

class ResnetClassifierModelTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
        self.BATCH_SIZE_TRAINING = 256
        self.BATCH_SIZE_VALIDATION = 256
        self.BATCH_SIZE_TESTING = 1

    def train(self,input_data, model_candidate_folder):
        #Prepare data 
        train_datagen = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                featurewise_center=True, 
                                                featurewise_std_normalization=True,
                                                zca_whitening=True,
                                                rotation_range=90,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                validation_split=0.2) # set validation split

        train_generator = train_datagen.flow_from_directory(
                                        os.path.join(input_data, "train/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TRAINING,
                                        class_mode='categorical',
                                        subset='training') # set as training data

        counter = Counter(train_generator.classes) 
        
        max_val = float(max(counter.values()))

        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 

        validation_generator = train_datagen.flow_from_directory(
                                os.path.join(input_data, "train/"), # same directory as training data
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_VALIDATION,
                                class_mode='categorical',
                                subset='validation') # set as validation data

        #instance models
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(50,50,3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000,activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        pred = tf.keras.layers.Dense(2,activation="softmax")(x)
        classifier = tf.keras.models.Model(inputs=base_model.input, outputs=pred)
        #disc, classifier = disc_network()

        #cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        cb_checkpointer = ModelCheckpoint(filepath = classifier_file, monitor = 'val_accuracy', save_best_only = True, mode = 'auto')

        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        classifier.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])

        _steps_per_epoch_training = len(train_generator.filenames)//self.BATCH_SIZE_TRAINING
        _steps_per_epoch_validation = len(validation_generator.filenames)//self.BATCH_SIZE_VALIDATION

        fit_history = classifier.fit_generator(
                                    train_generator,
                                    steps_per_epoch=_steps_per_epoch_training,
                                    epochs = self.epochs,
                                    validation_data=validation_generator,
                                    validation_steps=_steps_per_epoch_validation,
                                    class_weight=class_weights,
                                    callbacks=[cb_checkpointer])
        
        #load best weights
        classifier.load_weights(classifier_file)

        test_datagen = ImageDataGenerator(rescale=1./255) 
        test_generator = test_datagen.flow_from_directory(
                                        os.path.join(input_data, "eval/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TESTING,
                                        class_mode='categorical') # set as training data

        steps = len(test_generator.filenames)/self.BATCH_SIZE_TESTING
        result = classifier.predict_generator(test_generator,verbose=1,steps=steps)

        predicted_class_indices = np.argmax(result,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        #print("predictions :", predictions)
        y_true = test_generator.classes
        y_pred = np.argmax(result, axis=1)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1_s = f1_score(y_true, y_pred, average='macro')

        print("acc :", acc)
        print("recall : ", recall)
        print("precision : ", precision)
        print("f1_s :", f1_s)

        self.run.upload_file(name="diagnoz_classifier", path_or_stream=classifier_file)
        return classifier    
    #

class SimpleClassifierModelTrainerKeras(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
        self.BATCH_SIZE_TRAINING = 256
        self.BATCH_SIZE_VALIDATION = 256
        self.BATCH_SIZE_TESTING = 1

    def train(self,input_data, model_candidate_folder):
        #Prepare data 
        train_datagen = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                featurewise_center=True, 
                                                featurewise_std_normalization=True,
                                                zca_whitening=True,
                                                rotation_range=90,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                validation_split=0.2) # set validation split

        train_generator = train_datagen.flow_from_directory(
                                        os.path.join(input_data, "train/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TRAINING,
                                        class_mode='categorical',
                                        subset='training') # set as training data
        
        validation_generator = train_datagen.flow_from_directory(
                                os.path.join(input_data, "train/"), # same directory as training data
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_VALIDATION,
                                class_mode='categorical',
                                subset='validation') # set as validation data

        #instance models 
        disc, classifier = disc_network()

        #cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        cb_checkpointer = ModelCheckpoint(filepath = classifier_file, monitor = 'val_accuracy', save_best_only = True, mode = 'auto')

        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        classifier.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])

        _steps_per_epoch_training = len(train_generator.filenames)//self.BATCH_SIZE_TRAINING
        _steps_per_epoch_validation = len(validation_generator.filenames)//self.BATCH_SIZE_VALIDATION

        fit_history = classifier.fit_generator(
                                    train_generator,
                                    steps_per_epoch=_steps_per_epoch_training,
                                    epochs = self.epochs,
                                    validation_data=validation_generator,
                                    validation_steps=_steps_per_epoch_validation,
                                    callbacks=[cb_checkpointer])
        

        #load best weights
        classifier.load_weights(classifier_file)

        test_datagen = ImageDataGenerator(rescale=1./255) 
        test_generator = test_datagen.flow_from_directory(
                                        os.path.join(input_data, "eval/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TESTING,
                                        class_mode='categorical') # set as training data

        steps = len(test_generator.filenames)/self.BATCH_SIZE_TESTING
        result = classifier.predict_generator(test_generator,verbose=1,steps=steps)

        predicted_class_indices = np.argmax(result,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        #print("predictions :", predictions)
        y_true = test_generator.classes
        y_pred = np.argmax(result, axis=1)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1_s = f1_score(y_true, y_pred, average='macro')

        print("acc :", acc)
        print("recall : ", recall)
        print("precision : ", precision)
        print("f1_s :", f1_s)

        self.run.upload_file(name="diagnoz_classifier", path_or_stream=classifier_file)
        return classifier

class BalancedClassifierModelTrainerKeras(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
        self.BATCH_SIZE_TRAINING = 256
        self.BATCH_SIZE_VALIDATION = 256
        self.BATCH_SIZE_TESTING = 1
    
    def train(self,input_data, model_candidate_folder):
        #Prepare data 
        train_datagen = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                featurewise_center=True, 
                                                featurewise_std_normalization=True,
                                                zca_whitening=True,
                                                rotation_range=90,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                validation_split=0.2) # set validation split

        train_generator = train_datagen.flow_from_directory(
                                        os.path.join(input_data, "train/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TRAINING,
                                        class_mode='categorical',
                                        subset='training') # set as training data

        counter = Counter(train_generator.classes) 
        
        max_val = float(max(counter.values()))

        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 

        validation_generator = train_datagen.flow_from_directory(
                                os.path.join(input_data, "train/"), # same directory as training data
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_VALIDATION,
                                class_mode='categorical',
                                subset='validation') # set as validation data

        #instance models 
        disc, classifier = disc_network()

        #cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        cb_checkpointer = ModelCheckpoint(filepath = classifier_file, monitor = 'val_accuracy', save_best_only = True, mode = 'auto')

        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        classifier.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])

        _steps_per_epoch_training = len(train_generator.filenames)//self.BATCH_SIZE_TRAINING
        _steps_per_epoch_validation = len(validation_generator.filenames)//self.BATCH_SIZE_VALIDATION

        fit_history = classifier.fit_generator(
                                    train_generator,
                                    steps_per_epoch=_steps_per_epoch_training,
                                    epochs = self.epochs,
                                    validation_data=validation_generator,
                                    validation_steps=_steps_per_epoch_validation,
                                    class_weight=class_weights,
                                    callbacks=[cb_checkpointer])
        
        #load best weights
        classifier.load_weights(classifier_file)

        test_datagen = ImageDataGenerator(rescale=1./255) 
        test_generator = test_datagen.flow_from_directory(
                                        os.path.join(input_data, "eval/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TESTING,
                                        class_mode='categorical') # set as training data

        steps = len(test_generator.filenames)/self.BATCH_SIZE_TESTING
        result = classifier.predict_generator(test_generator,verbose=1,steps=steps)

        predicted_class_indices = np.argmax(result,axis=1)
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        #print("predictions :", predictions)
        y_true = test_generator.classes
        y_pred = np.argmax(result, axis=1)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        f1_s = f1_score(y_true, y_pred, average='macro')

        print("acc :", acc)
        print("recall : ", recall)
        print("precision : ", precision)
        print("f1_s :", f1_s)

        self.run.upload_file(name="diagnoz_classifier", path_or_stream=classifier_file)
        return classifier

class SimpleClassifierModelTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
    @tf.function
    def _train_classifier(self, images, labels, classifier, classifier_loss_fn):
        with tf.GradientTape() as tape:
            logits = classifier(images)
            classification_loss = classifier_loss_fn(labels, logits)
            
        grads = tape.gradient(classification_loss, classifier.trainable_weights)
        
        return classification_loss, grads

    def _check_point(self, classifier, model_candidate_folder):
        """ Save classifier and generator
        """
        os.makedirs(model_candidate_folder, exist_ok = True)
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        classifier.save(classifier_file)

    def _upaload_model(self, model_candidate_folder):
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        self.run.upload_file(name="diagnoz_classifier", path_or_stream=classifier_file)

    def train(self,input_data, model_candidate_folder):
        #Prepare data 
        train_datagen = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                featurewise_center=True, 
                                                featurewise_std_normalization=True,
                                                zca_whitening=True,
                                                rotation_range=90,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                validation_split=0.2) # set validation split

        labeled_subset = train_datagen.flow_from_directory(
                                        os.path.join(input_data, "train/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TRAINING_LABELED_SUBSET,
                                        class_mode='categorical',
                                        subset='training') # set as training data
        
        x_batch, y_batch = next(labeled_subset)

        #instance models 
        #generator = generator_network(latent_dim=128)
        disc, classifier = disc_network()


        assert classifier(np.expand_dims(x_batch[0], 0)).shape == (1, 2)

        """Define the optimizers
        """
        c_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        """Define the loss functions
        """
        classifier_loss_fn = tf.keras.losses.CategoricalCrossentropy()

        """The shared weights of the classifier and the discriminator would be updated on a set of 32 images:

            16 images from the set of only hundred labeled examples.
            8 images from the unlabeled examples.
            8 fake images generated by the generator.
        """

        c_loss = float("inf")
        ################## Training ##################
        ##############################################
        for epoch in tqdm(range(self.epochs)):
            """Define objects to calculate the mean losses across each epoch
            """
            c_loss_mean = tf.keras.metrics.Mean()
            
            """ Train the classifier
            for (images, labels) in labeled_subset:
            """
            images, labels = next(iter(labeled_subset)) # 16 images
            classification_loss, grads = self._train_classifier(images, labels, classifier, classifier_loss_fn)
            c_optimizer.apply_gradients(zip(grads, classifier.trainable_weights)) # The shared weights of the classifier and the discriminator will therefore be updated on a total of 32 images
            c_loss_mean.update_state(classification_loss)
                
            if epoch % 100 == 0:
                print("epoch: {} classification loss: {:.3f}".format(
                    epoch,
                    c_loss_mean.result()
                ))

            if c_loss > c_loss_mean.result():
                c_loss = c_loss_mean.result()
                print("\ncheck point with loss :", c_loss)
                self._check_point(classifier, model_candidate_folder)

        """ Load the best model
        """
        classifier.load_weights(os.path.join(model_candidate_folder, "classifier.hdf5"))

        """ Save classifier and generator
        """
        self._upaload_model(model_candidate_folder)

        return classifier

class ModelTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()

    @tf.function
    def _train_classifier(self, images, labels, classifier, classifier_loss_fn):
        with tf.GradientTape() as tape:
            logits = classifier(images)
            classification_loss = classifier_loss_fn(labels, logits)
            
        
        grads = tape.gradient(classification_loss, classifier.trainable_weights)
        
        return classification_loss, grads

    @tf.function
    def _train_disc(self, images, labels, disc_network, disc_loss_fn):
        with tf.GradientTape() as tape:
            logits = disc_network(images)
            disc_loss = disc_loss_fn(labels, logits)
        
        grads = tape.gradient(disc_loss, disc_network.trainable_weights)
        
        return disc_loss, grads

    @tf.function
    def _train_gen(self, random_latent_vectors, labels, disc_network, generator, gan_loss_fn):
        with tf.GradientTape() as tape:
            logits = disc_network(generator(random_latent_vectors))
            g_loss = gan_loss_fn(labels, logits)

        grads = tape.gradient(g_loss, generator.trainable_weights)
        
        return g_loss, grads

    def _check_point(self, classifier, generator,  model_candidate_folder):
        """ Save classifier and generator
        """
        os.makedirs(model_candidate_folder, exist_ok = True)
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        classifier.save(classifier_file)
        generator_file = os.path.join(model_candidate_folder, "generator.hdf5")
        generator.save(generator_file)

    def _upaload_model(self, model_candidate_folder):
        classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
        self.run.upload_file(name="diagnoz_classifier", path_or_stream=classifier_file)
        generator_file = os.path.join(model_candidate_folder, "generator.hdf5")
        self.run.upload_file(name="diagnoz_generator", path_or_stream=generator_file)

    def train(self,input_data, model_candidate_folder):
        #Prepare data 
        train_datagen = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                featurewise_center=True, 
                                                featurewise_std_normalization=True,
                                                zca_whitening=True,
                                                rotation_range=90,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                validation_split=0.2) # set validation split

        labeled_subset = train_datagen.flow_from_directory(
                                        os.path.join(input_data, "train/"),
                                        target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                        batch_size=self.BATCH_SIZE_TRAINING_LABELED_SUBSET,
                                        class_mode='categorical',
                                        subset='training') # set as training data

        unlabeled_dataset = train_datagen.flow_from_directory(
                                os.path.join(input_data, "unlabeled/"),
                                target_size=(self.IMAGE_RESIZE, self.IMAGE_RESIZE),
                                batch_size=self.BATCH_SIZE_TRAINING_UNLABELED_SUBSET,
                                class_mode=None,
                                subset='training')
        
        x_batch, y_batch = next(labeled_subset)

        #instance models 
        generator = generator_network(latent_dim=128)
        disc, classifier = disc_network()


        assert classifier(np.expand_dims(x_batch[0], 0)).shape == (1, 2)
        assert disc(np.expand_dims(x_batch[0], 0)).shape == (1, 1)
        assert generator(np.random.normal(size=(1, 128))).shape == (1, 50, 50, 3)

        """Define the optimizers
        """
        c_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        g_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        """Define the loss functions
        """
        classifier_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        disc_loss_fn = tf.keras.losses.BinaryCrossentropy()
        gan_loss_fn = tf.keras.losses.BinaryCrossentropy()

        """The shared weights of the classifier and the discriminator would be updated on a set of 32 images:

            16 images from the set of only hundred labeled examples.
            8 images from the unlabeled examples.
            8 fake images generated by the generator.
        """

        c_loss = float("inf")
        ################## Training ##################
        ##############################################
        for epoch in tqdm(range(self.epochs)):
        #for epoch in tqdm(range(7500)):
            """Define objects to calculate the mean losses across each epoch
            """
            c_loss_mean = tf.keras.metrics.Mean()
            d_loss_mean = tf.keras.metrics.Mean()
            g_loss_mean = tf.keras.metrics.Mean()
            
            """ Train the classifier
            for (images, labels) in labeled_subset:
            """
            images, labels = next(iter(labeled_subset)) # 16 images
            classification_loss, grads = self._train_classifier(images, labels, classifier, classifier_loss_fn)
            c_optimizer.apply_gradients(zip(grads, classifier.trainable_weights)) # The shared weights of the classifier and the discriminator will therefore be updated on a total of 32 images
            c_loss_mean.update_state(classification_loss)
            
            """
            ## Train discriminator and generator ##
            #######################################
            Train discriminator
            """
            real_images = next(iter(unlabeled_dataset)) # 8 real images
            batch_size = tf.shape(real_images)[0]
            random_latent_vectors = tf.random.normal(shape=(batch_size, 128))
            
            generated_images = generator(random_latent_vectors) # 8 fake images
            combined_images = tf.concat([generated_images, real_images], axis=0) # 16 total images
            combined_labels = tf.concat(
                [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
            ) # 0 -> Fake images, 1 -> Real images

            disc_loss, grads = self._train_disc(combined_images, combined_labels, disc, disc_loss_fn)
            d_optimizer.apply_gradients(zip(grads, disc.trainable_weights)) # The shared weights of the classifier and the discriminator will therefore be updated on a total of 32 images
            d_loss_mean.update_state(disc_loss)

            # Train the generator via signals from the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size*4, 128)) # 32 images
            misleading_labels = tf.ones((batch_size*4, 1))

            g_loss, grads = self._train_gen(random_latent_vectors, misleading_labels, disc, generator, gan_loss_fn)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
            g_loss_mean.update_state(g_loss)
                
            if epoch % 100 == 0:
                print("epoch: {} classification loss: {:.3f} dicriminator loss: {:.3f} gan loss:{:.3f}".format(
                    epoch,
                    c_loss_mean.result(),
                    d_loss_mean.result(),
                    g_loss_mean.result()
                ))

            if c_loss > c_loss_mean.result():
                c_loss = c_loss_mean.result()
                print("\ncheck point with loss :", c_loss)
                self._check_point(classifier, generator, model_candidate_folder)

        """ Load the best model
        """
        classifier.load_weights(os.path.join(model_candidate_folder, "classifier.hdf5"))

        """ Save classifier and generator
        """
        self._upaload_model(model_candidate_folder)

        return classifier
