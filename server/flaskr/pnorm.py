
def batch_norm(arcface_model, app):
    gamma = arcface_model.layers[app.config["ARCFACE_POOLING_LAYER_INDEX"]].weights[0]
    beta = arcface_model.layers["ARCFACE_POOLING_LAYER_INDEX"].weights[1]
    moving_mean = arcface_model.layers["ARCFACE_POOLING_LAYER_INDEX"].weights[2]
    moving_variance = arcface_model.layers["ARCFACE_POOLING_LAYER_INDEX"].weights[3]

def eval():
    