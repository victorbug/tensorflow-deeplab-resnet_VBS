"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function
print("Esta en inference.py")
import argparse
from datetime import datetime
import os
print("borrame1")
import sys
print("borrame2")
import time
print("borrame3")

from PIL import Image
print("borrame4")
import tensorflow as tf
print("borrame5")
import numpy as np
print("borrame6")

#from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, dense_crf, prepare_label
from deeplab_resnet import DeepLabResNetModel#Esta linea gatilla que se vaya a deeplab_resnet/init.py
print("Paso el import de DeepLabResNetModel")
from deeplab_resnet import ImageReader, decode_labels, dense_crf, prepare_label
print("Paso el import de ImageReader")



SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def get_arguments():
    print("Esta en inference.py/get_arguments")
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    print("Esta en inference.py/load")
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    print("Esta en inference.py/main")
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Prepare image.
    img_orig = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img_orig)
    img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    
    # CRF.
    print("Aca esta el CRF de /inference.py")
    raw_output_up = tf.nn.softmax(raw_output_up)    
    raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_orig, dim=0)], tf.float32)
    
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # Perform inference.
    preds = sess.run(pred)
    
    msk = decode_labels(preds)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    from datetime import datetime#VBS
    tiempo=datetime.today().strftime('%Y%m%d%H%M%S')#VBS
    pasos=str(dense_crf.__defaults__[1])#VBS. pasos se obtienen los parametros por default de otra funcion. Esta es una solución semi automatica
    im.save(args.save_dir + tiempo + '_' + pasos + '_mask.png')#Original modificada .    

    print('The output file has been saved to {}'.format(args.save_dir + tiempo + '_' + pasos + '_mask.png')) ##Original modificada    
    
    #from deeplab_resnet.utils.dense_crf import n_iters
    #import inspect
    #signature = inspect.signature(dense_crf)
    #print(dense_crf.__defaults__)
    #print(type(dense_crf.__defaults__))
    #print(dense_crf.__defaults__[1])
    #print(n_iters)    

    
if __name__ == '__main__':
    main()
