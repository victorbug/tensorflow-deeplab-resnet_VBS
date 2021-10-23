print("Esta en deeplab_resnet/init.py")
from .model import DeepLabResNetModel
from .image_reader import ImageReader
from .utils import decode_labels, dense_crf, inv_preprocess, prepare_label