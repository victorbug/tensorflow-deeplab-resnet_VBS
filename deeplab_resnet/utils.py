print("Esta en deeplab_resnet/utils.py")
from PIL import Image
import numpy as np
import tensorflow as tf
import pydensecrf.densecrf as dcrf

#n_classes = 21 #Ojo que esta es como una variable global dentro de este script y la parte delCRF lo usa harto
# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)    

def decode_labels(mask, num_images=1, num_classes=3):
    print("Esta en deeplab_resnet/utils.py/def decode_labels")
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes=3):
    print("Esta en deeplab_resnet/utils.py/def prepare_label")
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images=1):
  print("Esta en deeplab_resnet/utils.py/def inv_preprocess")
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
  for i in range(num_images):
    outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
  return outputs

def dense_crf(probs, img=None, n_iters=10, #Ojo originalmente es, n_iters=10
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC,
              num_classes=3):
    print("Esta en deeplab_resnet/utils.py/def dense_crf")
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    print("1 utils.py")
    _, h, w, _ = probs.shape
    print("2 utils.py: ", type(probs), probs.shape, "probs", probs[0][0][0][0])
    probs = probs[0].transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    print("3 utils.py: ", type(probs), probs.shape, "probs", probs[0][0][0])
    #print("DenseCRFVICTOR")
    d = dcrf.DenseCRF2D(w, h, num_classes) # Define DenseCRF model.
    print("4 utils.py:", type(d), d, "dcrf.DenseCRF2D(w, h, num_classes)")
    U = -np.log(probs)#+100*probs # Unary potential.
    #U = -np.log(probs)+100*probs # Unary potential.
    print("4 utils.py, U:", type(U),U.shape,U)
    print("5 utils.py: ", type(U),U.shape, "-np.log(probs)", U[0][0][0], -np.log(probs[0][0][0])+100*probs[0][0][0])
    U = U.reshape((num_classes, -1)) # Needs to be flat.
    print("6 utils.py: ", type(U), U.shape, "U.reshape((num_classes, -1))", U[0][0])
    d.setUnaryEnergy(U)#Aca se cae cuando se hace el cambio de numero de clases. d es un objeto del tipo dcrf.DenseCRF2D
    print("7 utils.py: ", type(d), d, "d.setUnaryEnergy(U)")
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    print("8 utils.py: ", type(d), d, "d.addPairwiseGaussian")
    if img is not None:
        print("9 utils.py: ", type(img),img.shape, "img", img[0][0][0][0])
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        print("10 utils.py: ")
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
        print("11: ", type(d), d, "d.addPairwiseBilateral")
    #print("10")
    Q = d.inference(n_iters)
    print("12 utils.py: ",type(Q), Q,"d.inference(n_iters)") 
    preds = np.array(Q, dtype=np.float32).reshape((num_classes, h, w)).transpose(1, 2, 0)
    print("13 utils.py: ", type(preds),preds.shape, "preds", preds[0][0][0])
    print("14 utils.py: ", "n_iters=",n_iters)
    return np.expand_dims(preds, 0)
        
    
              
