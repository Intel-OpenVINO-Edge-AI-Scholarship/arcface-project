import os
from flask import Flask

def create_app(test_config=None, dbfile=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, dbfile),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app

def create_keras_app(dbfile=None, modelfile=None, pnorm=None, lognorm=None):
    app = create_app(dbfile=dbfile)

    import numpy as np
    from keras.models import Sequential, load_model, Model
    from keras.datasets import mnist
    from metrics_face import ArcFace
    import tensorflow.compat.v1 as tf
    from glob import glob
    import os
    import facenet
    from keras.preprocessing import image
    from tqdm import tqdm
    import keras
    from PIL import Image
    import cv2

    arcface_model = load_model(modelfile, custom_objects={'ArcFace': ArcFace})

    arcface_model1 = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)
    
    return arcface_model1