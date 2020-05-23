import keras

class_weights = [1, 0.5, 1, 10, 10]
def intermediate_model(x):
    return (
        keras.activations.elu((x-keras.backend.mean(x, axis=1)) / keras.backend.square(keras.backend.std(x))
    ) + keras.backend.square(x-keras.backend.mean(x, axis=1)) / keras.backend.std(x)
    ) + keras.backend.min(x, axis=1) * keras.backend.log(keras.backend.square(x-keras.backend.mean(x, axis=1)))