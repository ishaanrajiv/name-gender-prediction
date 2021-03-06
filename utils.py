from itertools import product
from string import ascii_lowercase
import re,string
from keras.models import load_model
import numpy as np, pandas as pd
import math
import tensorflow as tf

class Utils:

    def __init__(self):
        self.pattern = re.compile('[^a-z ]+')
        self.flatten = lambda l: [item for sublist in l for item in sublist]
        self.max_name_length = 20
        self.min_name_length = 3
        self.wordPatternModelPath = "model/word-pattern-nn.hdf5"
        self.wordPatternModel = load_model("model/word-pattern-nn.hdf5")
        self.characterModelPath = "model/character-nn.hdf5"
        self.characterModel = load_model("model/character-nn.hdf5")
        self.trainPath = "dataset/train.csv"
        self.testPath = "dataset/test.csv"
        self.staticWordPatternDict = self.wordPatternDict()
        self.staticcharacterDict = self.characterDict()
        self.wordPatternLen = 8

    def characterDict(self):
        return dict(zip(list(self.pattern.sub('', string.printable)+'X'),range(27,-1,-1)))

    def characterEncodeNameString(self,name,encodeDict):
        name = str(name).strip().lower()
        return [encodeDict.get(y,0) for y in self.pattern.sub('',str(name)).ljust(self.max_name_length,'X')]

    def wordPatternDict(self):
        keywords = [''.join(i) for i in product(ascii_lowercase, repeat = 3)] + [''.join(i) for i in product(ascii_lowercase, repeat = 2)] + [''.join(i) for i in product(ascii_lowercase, repeat = 1)]
        encode_dict={'XYZ':0}
        i=1
        for word in keywords:
            encode_dict[word]=i
            i+=1
        return encode_dict

    def wordPatternEncodeNameString(self,name,encodeDict):
        name = self.pattern.sub('', str(name).strip().lower())
        name_arr = self.flatten([[a[0:3],a[-3:],a[0:2],a[-2:]] for a in name.split()[0:2]])
        final_arr = ['XYZ','XYZ','XYZ','XYZ','XYZ','XYZ','XYZ','XYZ']
        final_arr[:len(name_arr)]=name_arr
        return [encodeDict[y] for y in final_arr]

    def wordPatternGenderPredict(self,name,encodeDict,model):
        name=self.wordPatternEncodeNameString(name,encodeDict)
        dim = int(math.log(len(encodeDict),2))+1
        return model.predict(np.array([np.array([(((x & (1 << np.arange(dim)))) > 0).astype(int) for x in name])]))

    def newWordPatternGenderPredict(self,name,encodeDict,model):
        name=self.wordPatternEncodeNameString(name,encodeDict)
        dict_len = len(encodeDict)
        X = np.zeros((1,self.wordPatternLen, dict_len), dtype=np.bool)
        for t, phrase in enumerate(name):
                X[0, t, phrase] = 1
        return model.predict(X)

    def characterGenderPredict(self,name,encodeDict,model):
        name=self.characterEncodeNameString(name,encodeDict)
        dim = int(math.log(len(encodeDict),2))+1
        return model.predict(np.array([np.array([(((x & (1 << np.arange(dim)))) > 0).astype(int) for x in name])]))
