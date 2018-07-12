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

filepath = "model/character-nn.hdf5"
model = load_model(filepath)

### Config Parameters
min_name_length = 3 # Min length of name
max_name_length = 20
pattern = re.compile('[^a-z ]+')
encode_dict = dict(zip(list(pattern.sub('', string.printable)+'X'),range(27,-1,-1)))
char_dim = int(math.log(len(encode_dict),2))+1



def encodeNameString(name):
    pattern = re.compile('[^a-z ]+')
    encode_dict = dict(zip(list(pattern.sub('', string.printable)+'X'),range(27,-1,-1)))
    name = str(name).strip().lower()
    return [encode_dict.get(y,0) for y in pattern.sub('',str(name)).ljust(max_name_length,'X')]


def genderPredict(name):
    name=encodeNameString(name)
    return model.predict(np.array([np.array([(((x & (1 << np.arange(char_dim)))) > 0).astype(int) for x in name])]))


#def main():
while(1):
    try:
        name=input("Enter Name:")
    except Exception as e:
        print("Aborting...\n",str(e))
    gender = {0:'female',1:'male'}
    gender_arr = genderPredict(name)
    print(gender[np.argmax(gender_arr)],np.max(gender_arr))
