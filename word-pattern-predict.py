import pandas as pd
import numpy as np
import re, string
from keras.models import Sequential
from keras.regularizers import l2 # L2-regularisation
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import  LeakyReLU  
from keras.layers.normalization import BatchNormalization
from keras import layers
import math
from keras.models import load_model

filepath = "model/word-pattern-nn.hdf5"
model = load_model(filepath)

flatten = lambda l: [item for sublist in l for item in sublist]

from itertools import product
from string import ascii_lowercase
keywords = [''.join(i) for i in product(ascii_lowercase, repeat = 3)] + [''.join(i) for i in product(ascii_lowercase, repeat = 2)] + [''.join(i) for i in product(ascii_lowercase, repeat = 1)]
encode_dict={'XYZ':0}
i=1
for word in keywords:
    encode_dict[word]=i
    i+=1

char_dim = int(math.log(len(encode_dict),2))+1

## Cleaning Up
def encodeNameString(name):
    pattern = re.compile('[^a-z ]+')
    name = pattern.sub('', str(name).strip().lower())
    # name_arr = flatten([[a[0:3],a[-3:]] for a in name.split()[0:2]])
    # final_arr = ['XYZ','XYZ','XYZ','XYZ']
    name_arr = flatten([[a[0:3],a[-3:],a[0:2],a[-2:]] for a in name.split()[0:2]])
    final_arr = ['XYZ','XYZ','XYZ','XYZ','XYZ','XYZ','XYZ','XYZ']
    final_arr[:len(name_arr)]=name_arr
    return [encode_dict[y] for y in final_arr]


def genderPredict(name):
    name=encodeNameString(name)
    return model.predict(np.array([np.array([(((x & (1 << np.arange(char_dim)))) > 0).astype(int) for x in name])]))


#def main():
try:
    while(1):
        name=input("\nEnter Name:")
        gender = {0:'female',1:'male'}
        gender_arr = genderPredict(name)
        print(gender[np.argmax(gender_arr)],np.max(gender_arr))
except Exception as e:
    print("Aborting...\n",str(e))
    