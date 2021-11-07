print("Esta en kaffe/caffe/resolver.py")
import sys

SHARED_CAFFE_RESOLVER = None

class CaffeResolver(object):
    print("Esta en kaffe/caffe/resolver.py/Class CaffeResolver")
    def __init__(self):
        print("Esta en kaffe/caffe/resolver.py/Class CaffeResolver/def __init__")
        self.import_caffe()

    def import_caffe(self):
        print("Esta en kaffe/caffe/resolver.py/Class CaffeResolver/def import_caffe")
        self.caffe = None
        try:
            # Try to import PyCaffe first
            import caffe
            self.caffe = caffe
        except ImportError:
            # Fall back to the protobuf implementation
            from . import caffepb
            self.caffepb = caffepb
            show_fallback_warning()
        if self.caffe:
            # Use the protobuf code from the imported distribution.
            # This way, Caffe variants with custom layers will work.
            self.caffepb = self.caffe.proto.caffe_pb2
        self.NetParameter = self.caffepb.NetParameter

    def has_pycaffe(self):
        print("Esta en kaffe/caffe/resolver.py/Class CaffeResolver/def has_pycaffe")
        return self.caffe is not None

def get_caffe_resolver():
    print("Esta en kaffe/caffe/resolver.py/def get_caffe_resolver")
    global SHARED_CAFFE_RESOLVER
    if SHARED_CAFFE_RESOLVER is None:
        SHARED_CAFFE_RESOLVER = CaffeResolver()
    return SHARED_CAFFE_RESOLVER

def has_pycaffe():
    print("Esta en kaffe/caffe/resolver.py/def has_pycaffe")
    return get_caffe_resolver().has_pycaffe()

def show_fallback_warning():
    print("Esta en kaffe/caffe/resolver.py/def show_fallback_warning")
    msg = '''
------------------------------------------------------------
    WARNING: PyCaffe not found!
    Falling back to a pure protocol buffer implementation.
    * Conversions will be drastically slower.
    * This backend is UNTESTED!
------------------------------------------------------------

'''
    sys.stderr.write(msg)
