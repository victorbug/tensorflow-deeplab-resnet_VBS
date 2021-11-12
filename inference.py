"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function
print("Esta en inference.py")
import argparse
from datetime import datetime
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np

#from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, dense_crf, prepare_label
print("Esta en inference.py. 1) Va a hacer el from deeplab_resnet import de DeepLabResNetModel")
from deeplab_resnet import DeepLabResNetModel#Esta linea gatilla que se vaya a deeplab_resnet/init.py
print("Esta en inference.py. 2) Va a hacer el from deeplab_resnet import ImageReader, decode_labels, dense_crf, prepare_label")
from deeplab_resnet import ImageReader, decode_labels, dense_crf, prepare_label

NUM_CLASSES = 21
SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

print("Siguen las definiciones en el archivo inference.py")
def get_arguments():
    print("Esta en inference.py/def get_arguments")
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    print("Esta en inference.py/def load")
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
    print(saver)

def main():
    print("Esta en inference.py/def main")
    
    """Create the model and start the evaluation process."""
    print("1.- inference.py: Create the model and start the evaluation process.")
    args = get_arguments()
    
    # Prepare image.
    print("2.- inference.py: Prepare image.")
    img_orig = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)#Es como un ndarray de dimension de: ancho pixeles x alto pixeles x 3 colores. Si se printea el tf.read_file(), da caracteres extranos y muy extensos en el terminal
    #Si imprimo el valor de tf.read_file(args.img_path) salen caracteres muy raros
    
    #print(tf.read_file(args.img_path))
    #print(img_orig)
    #print(type(tf.read_file(args.img_path)))
    #print(type(img_orig))
    #print(img_orig[0][0])
   
    # Convert RGB to BGR.
    print("3.- inference.py: Convert RGB to BGR.")
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img_orig)#QUizas el split_dim significa que considera 2 dimensiones (ancho y alto) y en base a eso corta en 3
    img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)#tf.cast los transforma a enteros si se usa dtype=tf.int32
    #print(img_r)

    if True:
        with tf.Session() as sess:
            print(type(img_orig.eval()), img_orig.eval().shape)
            print(type(tf.read_file(args.img_path).eval()), "No imprimir el valor porque es muy extenso y caracteres raros")
            print(type(img_r.eval()), img_r.eval().shape)
            print(type(tf.concat(2, [img_b, img_g, img_r]).eval()), tf.concat(2, [img_b, img_g, img_r]).eval().shape) 
            print("Ejemplo juguete",tf.concat(0, [[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]]).eval())
            print(type(img.eval()), img.eval().shape)
            print("Ejemplo juguete", tf.expand_dims([2,2] , dim=1).eval())#A [2,2], dim=0 lo convierte en [[2 2]]. dim=1 lo convierte en [[2][2]]


    # Extract mean.
    print("4.- inference.py: Extract mean.")
    img -= IMG_MEAN #Este promedio esta fijo.....quizas debe adecuarse para cada imagen. A la variable img, se le resta la variable IMG_MEAN y se almacena en la variable img
    
    # Create network.
    print("5.- inference.py: Create network.")
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)#expands_dims le anade una dimension, nada muy especial, supongo que es por algo simple, como x ejemplo que la otra funcion solo recibe variables con esas dimensiones
    print("net",type(net), net)

    # Which variables to load.
    print("6.- inference.py: Which variables to load.")
    restore_var = tf.global_variables()
    print("restore_var", len(restore_var), restore_var[0], "es una lista de muchas variables, mejor solo imprimir el primer elemento, no la lista completa")

    # Predictions.
    print("7.- inference.py: Predictions.")
    raw_output = net.layers['fc1_voc12']#Es objeto Tensor
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])

    print(raw_output, type(raw_output), "No me deja hacerle .eval(). Dice: FailedPreconditionError (see above for traceback): Attempting to use uninitialized value bn2c_branch2a/moving_variance")

    #with tf.Session() as sess2:
        #result9 = raw_output.eval()
    #    print(1)
    #print(result9)

    # CRF.
    print("8.- inference.py: CRF.")#crear un raw_output_up que sea nulo creo que es sinonimo de independizar el bloque CRF de lo anterior
    raw_output_up = tf.nn.softmax(raw_output_up)    
    raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_orig, dim=0)], tf.float32)
    
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    #print(pred.shape)
    
    # Set up TF session and initialize variables. 
    print("9.- inference.py: Set up TF session and initialize variables.")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    print("10.- inference.py: Load weights.")
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # Perform inference.
    print("11.- inference.py: Perform inference.")
    #if True:
    #    with tf.Session() as sess2:
    #        print(type(pred.eval()))
    #print(pred)

    #with tf.Session() as sess:
    #    print(pred.eval())

    preds = sess.run(pred)#Aca se cae si le cambio el numero de clases
    print("preds (ojo las dimensiones tienen que ver con los pixeles de la imagen (los valores del medio del vector de 4 elementos))", type(preds), preds.shape, preds[0].shape, preds[0][0].shape, preds[0][0][0].shape, preds[0][0][0][0].shape, preds[0][0][0][0].shape)
    print(preds[0][0][0][0], preds[0][0][0], "Printear preds[0][0], ya deja un vector muy largo")

    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    from datetime import datetime#VBS
    tiempo=datetime.today().strftime('%Y%m%d%H%M%S')#VBS
    pasos=str(dense_crf.__defaults__[1])#VBS. pasos se obtienen los parametros por default de otra funcion. Esta es una solucion semi automatica
    im.save(args.save_dir + tiempo + '_niters_' + pasos + '_mask.png')#Original modificada .    

    print('The output file has been saved to {}'.format(args.save_dir + tiempo + '_niters_' + pasos + '_mask.png')) ##Original modificada    
    
    #from deeplab_resnet.utils.dense_crf import n_iters
    #import inspect
    #signature = inspect.signature(dense_crf)
    #print(dense_crf.__defaults__)
    #print(type(dense_crf.__defaults__))
    #print(dense_crf.__defaults__[1])
    #print(n_iters)    

    
if __name__ == '__main__':
    main()
