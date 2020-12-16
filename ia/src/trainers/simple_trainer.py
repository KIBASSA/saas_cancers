from generator import generator_network # pylint: disable=no-name-in-module
from discriminator import disc_network # pylint: disable=import-error
from global_helpers import AzureMLLogsProvider # pylint: disable=import-error
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
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from random import shuffle
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
