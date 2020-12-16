import os
import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from abstract_trainer import AbstractModelTrainer

class SupervisedContrastiveLearning(AbstractModelTrainer):
    
    def train(self,input_data, model_candidate_folder, loader_data):
        pattern_images = os.path.join(input_data,'train/**/*.png')
        result = loader_data.load(pattern_images)
        X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot = result

        #Using image data augmentation
        data_augmentation = keras.Sequential(
        [
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.02),
                layers.experimental.preprocessing.RandomWidth(0.2),
                layers.experimental.preprocessing.RandomHeight(0.2),
            ]
        )

        # Setting the state of the normalization layer.
        data_augmentation.layers[0].adapt(x_train)

        #Build the encoder model