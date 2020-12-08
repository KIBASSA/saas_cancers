import horovod.tensorflow as hvd
from generator import generator_network
from discriminator import disc_network
from global_helpers import AzureMLLogsProvider
from abstract_trainer import AbstractModelTrainer, ModelTrainerType, ModelTrainerFactory
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tqdm import tqdm
from enum import Enum

class GanDistributedClassifierModelTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()
    
    @tf.function
    def _train_classifier(self, images, labels, classifier, classifier_loss_fn):
        with tf.GradientTape() as tape:
            logits = classifier(images)
            classification_loss = classifier_loss_fn(labels, logits)
        
        #Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(classification_loss, classifier.trainable_weights)
        return classification_loss, grads

    @tf.function
    def _train_disc(self, images, labels, disc_network, disc_loss_fn):
        with tf.GradientTape() as tape:
            logits = disc_network(images)
            disc_loss = disc_loss_fn(labels, logits)
        
        #Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(disc_loss, disc_network.trainable_weights)
        
        return disc_loss, grads

    @tf.function
    def _train_gen(self, random_latent_vectors, labels, disc_network, generator, gan_loss_fn):
        with tf.GradientTape() as tape:
            logits = disc_network(generator(random_latent_vectors))
            g_loss = gan_loss_fn(labels, logits)

        #Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(g_loss, generator.trainable_weights)
        
        return g_loss, grads

    def train(self,input_data, model_candidate_folder):

        # Initialize Horovod
        hvd.init()

        # Pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

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
        c_optimizer = tf.keras.optimizers.Adam(lr=0.0002 * hvd.size(), beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(lr=0.0002 * hvd.size(), beta_1=0.5)
        g_optimizer = tf.keras.optimizers.Adam(lr=0.0002 * hvd.size(), beta_1=0.5)

        """Define the loss functions
        """
        classifier_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        disc_loss_fn = tf.keras.losses.BinaryCrossentropy()
        gan_loss_fn = tf.keras.losses.BinaryCrossentropy()

        checkpoint_dir = './checkpoints'
        checkpoint = tf.train.Checkpoint(model=classifier, optimizer=c_optimizer)

        """The shared weights of the classifier and the discriminator would be updated on a set of 32 images:

            16 images from the set of only hundred labeled examples.
            8 images from the unlabeled examples.
            8 fake images generated by the generator.
        """

        c_loss = float("inf")
        ################## Training ##################
        ##############################################
        for epoch in tqdm(range(self.epochs // hvd.size())):
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

            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.
            if epoch == 0:
                #classifier
                hvd.broadcast_variables(classifier.variables, root_rank=0)
                hvd.broadcast_variables(c_optimizer.variables(), root_rank=0)

                #disc
                hvd.broadcast_variables(disc.variables, root_rank=0)
                hvd.broadcast_variables(d_optimizer.variables(), root_rank=0)

                #disc
                hvd.broadcast_variables(generator.variables, root_rank=0)
                hvd.broadcast_variables(g_optimizer.variables(), root_rank=0)



        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting it.
        if hvd.rank() == 0:
            checkpoint.save(checkpoint_dir)
        """ Save classifier and generator
        """
        self._upaload_model(model_candidate_folder)

        return classifier
 