print("Esta en kaffe/errors.py")
import sys

class KaffeError(Exception):	
    pass
    #print("Esta en kaffe/errors.py/Class KaffeError")

def print_stderr(msg):
	#print("Esta en kaffe/errors.py/Class KaffeError/print_stderr")
    sys.stderr.write('%s\n' % msg)
