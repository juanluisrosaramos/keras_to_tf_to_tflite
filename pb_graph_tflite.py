import tensorflow as tf
import numpy as np
from pathlib import Path
from absl import flags
from absl import app
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', None, 'Path to the input model.')
flags.DEFINE_string('input_arrays', None, 'Name of the input layer')
flags.DEFINE_string('output_arrays', None, 'Name of the output layer')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_string('input_shape', None, 'Shape of the inputs')

flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')

def convert(input_model,input_arrays,output_arrays,input_shape, output_model):
    # Converting a GraphDef from file.    
    input_arrays = [input_arrays]
    output_arrays = [output_arrays]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        input_model, input_arrays, output_arrays,input_shapes={"input":input_shape})
    tflite_model = converter.convert()    
    open(output_model, "wb").write(tflite_model)

def test(output_model):
    # Load TFLite model and allocate tensors. 
    #    
    interpreter = tf.lite.Interpreter(model_path=str(output_model))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print('---------------------------------------------')
    print(output_details)

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)

def main(args):
    ##Output model
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent    
    output_model_name = Path(output_model).name
    output_model_pbtxt_name = output_model_name + '.tflite'
    output_model_name = Path(output_fld,output_model_pbtxt_name)
    print(output_model_name)
    # Create output directory if it does not exist
    Path(output_fld).parent.mkdir(parents=True, exist_ok=True)
    
    ##convert shapes args
    input_shape = list(FLAGS.input_shape.split(","))    
    convert(FLAGS.input_model,FLAGS.input_arrays,FLAGS.output_arrays,input_shape,output_model_name)
    
    logging.info('Saved the tflite export at %s',
                 str(output_model_name))
    test(output_model_name)

if __name__ == "__main__":
    app.run(main)