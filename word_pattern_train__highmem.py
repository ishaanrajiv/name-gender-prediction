import pandas as pd
import numpy as np
import re, string
from keras.models import Sequential
from keras.regularizers import l2 # L2-regularisation
from keras.layers import Dense, Activation, Dropout
from keras import layers
import math
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint

from utils import Utils

util = Utils()
encode_dict = util.wordPatternDict()

def cleanName(dataframe):
    global util,encode_dict
    dataframe = dataframe['Name']
    dataframe = dataframe.apply(lambda x: util.wordPatternEncodeNameString(x,encode_dict))
    return pd.DataFrame(dataframe)


def kerasModel(input_dim,output_dim,dict_len):
    model = Sequential()
    model.add(Dense(128, input_shape=(8,dict_len), kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.3)))
    model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(30, init='uniform', activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    
    global util,encode_dict

    df = pd.read_csv(util.trainPath)   
    df = df[df['Name'].map(str).map(len) >= util.min_name_length][df['Name'].map(str).map(len) <= util.max_name_length]

    df_X = cleanName(df)

    for i in range(0,8):
        df_X[i]=df_X['Name'].apply(lambda x: x[i])

    df_X = df_X.drop(columns=['Name'])
    df_Y = pd.get_dummies(df['Gender'])
    
    y = df_Y.values

    X = np.zeros((len(df_X), 8, len(encode_dict)), dtype=np.bool)
    for i, name in enumerate(df_X.values):
        for t, phrase in enumerate(name):
            X[i, t, phrase] = 1

    input_dim = len(X[0])
    output_dim = len(y[0])

    checkpoint = ModelCheckpoint(util.wordPatternModelPath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max',
                                period=1)


    model = kerasModel(input_dim,output_dim,len(encode_dict))

    model.fit(X, y, epochs=50, batch_size=100,  verbose=1, validation_split=0.2, shuffle=True,callbacks=[checkpoint])

    model.save(filepath)

