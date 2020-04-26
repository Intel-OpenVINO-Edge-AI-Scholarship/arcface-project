import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler

import archs
from metrics import *

def obtain_model(modelpath):
    arcface_model = load_model(modelpath, custom_objects={'ArcFace': ArcFace})
    arcface_model = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)

    return arcface_model