import torch
from torch import nn
import tensorflow as tf
import argparse
from keras import backend as K
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_filename", default=None, 
        help='conversion protocol buffer filename', required=True)
    parser.add_argument("--pb_filename_text", default=None, 
        help='conversion protocol buffer filename', required=True)
    parser.add_argument('--arch', '-a', metavar='ARCH', default=None, required=True)
    parser.add_argument('--num-features', default=5, type=int,
        help='dimention of embedded features', required=True)

    return parser.parse_args()

def create_model(args):
    import arcface
    return arcface.obtain_arcface_model(args)

def freeze_session(session, keep_var_names=None, keep_output_names=None, 
output_names=None, clear_devices=None):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        mod_graph_def = tf.GraphDef()
        nodes = []
        remove_nodes = ['arc_face_1/arcface_acos', 'arc_face_1/arcface_cos']
        if clear_devices:
            for node in input_graph_def.node:
                if node.name not in remove_nodes:
                    nodes.append(node)
                node.device = ""
            mod_graph_def.node.extend(nodes)
            # Delete references to deleted nodes
            for node in mod_graph_def.node:
                inp_names = []
                for inp in node.input:
                    if (remove_nodes[0] in inp) or (remove_nodes[1] in inp):
                        pass
                    else:
                        inp_names.append(inp)

                del node.input[:]
                node.input.extend(inp_names)
        print("Removed nodes: ", ', '.join(remove_nodes))
        frozen_graph = convert_variables_to_constants(session, mod_graph_def, 
        output_names, freeze_var_names)

        return frozen_graph

if __name__ == "__main__":

    args = parse_args()
    model = create_model(args)

    frozen_graph = freeze_session(K.get_session(),
        output_names=[out.op.name for out in model.outputs], clear_devices=True)
    tf.train.write_graph(frozen_graph, "models/mnist_vgg8_arcface_5d", name=args.pb_filename, as_text=False)
    tf.train.write_graph(frozen_graph, "models/mnist_vgg8_arcface_5d", name=args.pb_filename_text, as_text=True)
