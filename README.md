# Keras to TensorFlow to tfLite
This part is created by [Amir Abdi](https://github.com/amir-abdi/keras_to_tensorflow). I added my part for converting to tflite from the binary protobuf.

The **keras_to_tensorflow** is a tool that converts a trained keras model into a ready-for-inference TensorFlow model. *The tool is NOT tailored for TensorFlow 2.0 it SEEMS to be working fine.*

#### Summary
- In the default behaviour, this tool **freezes** the nodes (converts all TF variables to TF constants), and saves the inference graph and weights into a binary protobuf (.pb) file. During freezing, TensorFlow also applies node pruning which removes nodes with no contribution to the output tensor.

- This tool supports multiple output networks and enables the user to rename the output tensors via the `--output_nodes_prefix` flag.
 
- If the `--output_meta_ckpt` flag is set, the checkpoint and metagraph files for TensorFlow will also be exported
which can later be used in the `tf.train.Saver` class to continue training.   


## How to use Keras to Binary Graph
Keras models can be saved as a single [`.hdf5` or `h5`] file, which stores both the architecture and weights, using the `model.save()` function.
 This model can be then converted to a TensorFlow model by calling this tool as follows:
    
    python keras_to_tensorflow.py 
        --input_model="path/to/keras/model.h5" 
        --output_model="path/to/save/model.pb"
     
Keras models can also be saved in two separate files where a [`.hdf5` or `h5`] file stores the weights, using the `model.save_weights()` function, and another `.json` file stores the network architecture using the `model.to_json()` function.
In this case, the model can be converted as follows:

    python keras_to_tensorflow.py 
        --input_model="path/to/keras/model.h5" 
        --input_model_json="path/to/keras/model.json" 
        --output_model="path/to/save/model.pb"

Try 
    python keras_to_tensorflow.py --help
to learn about other supported flags (quantize, output_nodes_prefix, save_graph_def).

## How to use PB to tfLite
As Keras to tfLite failed quite times we suggested this step in the pipeline. While you use the previous transformer we suggest to take note of input layer name, output and also inputs shapes (sizes in the case of images)
    
    python pb_graph_tflite.py 
        --input_model="models/wpod-net_update1.pb" 
        --input_arrays="input"
        --output_arrays="concatenate_1/concat"
        --output_model="models/tflite_converted"
        --input_shape="1,240,80,3"


## Dependencies
- keras
- tensorflow
- absl
- pathlib

## Legacy code
The code on how to freeze and save keras models in previous versions of tensorflow is also available. Back then, the freeze_graph tool (```/tensorflow/python/tools/freeze_graph.py```) was used to convert the variables into constants. This functionality is now handled by ```graph_util.convert_variables_to_constants```
