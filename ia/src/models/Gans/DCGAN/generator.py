import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
# Generator
# Reference: Chapter 20, GANs in Python by Jason Brownlee, Chapter 7 of GANs in Action
def generator_network(latent_dim):
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 256 * 3 * 3
    gen = Dense(n_nodes)(in_lat)
    gen = Reshape((3, 3, 256))(gen)

    gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
    # gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.01)(gen)

    gen = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(gen)
    # gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.01)(gen)
    
    gen = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(gen)
    # gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.01)(gen)
    
    gen = Conv2DTranspose(16, (3,3), strides=(1,1), padding='same')(gen)
    # gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.01)(gen)
    
    gen = Conv2DTranspose(8, (3,3), strides=(2,2), padding='same')(gen)
    # gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.01)(gen)
    
    gen = ZeroPadding2D()(gen)
    
    out_layer = Conv2DTranspose(3, (3,3), strides=(1, 1), activation='tanh', padding='same')(gen)

    model = Model(in_lat, out_layer)

    return model