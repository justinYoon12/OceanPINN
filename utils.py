import numpy as np
import random 
import tensorflow as tf
import os
from os.path import dirname, join as pjoin
import io
import json

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def writejson(directory,data):
    json_object = json.dumps(data,indent=4)
    with open(pjoin(directory,'data.json'), "w") as outfile:
        # json.dumps(data,outfile,indent=4, sort_keys=True, separators=(',', ': '))
        outfile.write(json_object)
