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
encode_dict = util.characterDict()

def cleanName(dataframe):
    global util,encode_dict
    dataframe = dataframe['Name']
    dataframe = dataframe.apply(lambda x: util.characterEncodeNameString(x,encode_dict))
    return pd.DataFrame(dataframe)


def kerasModel(input_dim,output_dim,char_dim):
    model = Sequential()
    model.add(Dense(input_dim+1, input_shape=(input_dim,char_dim), kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.3)))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(3, init='uniform', activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    global util,encode_dict

    df = pd.read_csv(util.trainPath)   
    df = df[df['Name'].map(str).map(len) >= util.min_name_length][df['Name'].map(str).map(len) <= util.max_name_length]

    df_X = cleanName(df)

    for i in range(0,util.max_name_length):
        df_X[i]=df_X['Name'].apply(lambda x: x[i])

    df_X = df_X.drop(columns=['Name'])

    dim = int(math.log(len(encode_dict),2))+1
    X =np.array([(((x[:,None] & (1 << np.arange(dim)))) > 0).astype(int) for x in df_X.values])

    df_Y = pd.get_dummies(df['Gender'])
    y = df_Y.values

    input_dim = len(X[0])
    output_dim = len(y[0])

    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(util.characterModelPath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max',
                                period=1)


    model = kerasModel(input_dim,output_dim,dim)

    model.fit(X, y, epochs=5000000, batch_size=15000,  verbose=1, validation_split=0.2, shuffle=True,callbacks=[checkpoint])

    model.save(filepath)


if __name__ == "__main__":
    main()









# import pandas as pd
# import numpy as np
# import re, string
# from keras.models import Sequential
# from keras.regularizers import l2 # L2-regularisation
# from keras.layers import Dense, Activation, Dropout
# from keras.layers.advanced_activations import  LeakyReLU  
# from keras.layers.normalization import BatchNormalization
# from keras import layers
# import math
# from keras.layers import Flatten

# ### Config Parameters
# min_name_length = 3 # Min length of name
# max_name_length = 20

# ### Read Train Dataset - Name,Gender header
# df = pd.read_csv("dataset/train.csv")

# ### Remove names with less than 2 characters
# df = df[df['Name'].map(str).map(len) >= min_name_length][df['Name'].map(str).map(len) <= max_name_length]

# pattern = re.compile('[^a-z ]+')
# encode_dict = dict(zip(list(pattern.sub('', string.printable)+'X'),range(27,-1,-1)))
# char_dim = int(math.log(len(encode_dict),2))+1

# ## Cleaning Up
# def encodeNameString(name):
#     pattern = re.compile('[^a-z ]+')
#     encode_dict = dict(zip(list(pattern.sub('', string.printable)+'X'),range(27,-1,-1)))
#     name = str(name).strip().lower()
#     return [encode_dict.get(y,0) for y in pattern.sub('',str(name)).ljust(max_name_length,'X')]
#     # return np.array([(((x & (1 << np.arange(char_dim)))) > 0).astype(int) for x in name_arr])

# def cleanName(dataframe):
#     dataframe = dataframe['Name']
#     dataframe = dataframe.apply(lambda x: encodeNameString(x))
#     return pd.DataFrame(dataframe)



# df_X = cleanName(df)

# for i in range(0,max_name_length):
#     df_X[i]=df_X['Name'].apply(lambda x: x[i])

# # from functools import reduce
# # dfs = [pd.get_dummies(df_X['Name'].apply(lambda x: x[i])) for i in range(0,max_name_length)]
# # df_X = pd.concat(dfs, axis=1)

# df_X = df_X.drop(columns=['Name'])
# X = df_X.values

# X =np.array([(((x[:,None] & (1 << np.arange(char_dim)))) > 0).astype(int) for x in df_X.values])

# df_Y = pd.get_dummies(df['Gender'])
# y = df_Y.values

# input_dim = len(X[0])
# output_dim = len(y[0])

# model = Sequential()
# model.add(Dense(input_dim+1, input_shape=(input_dim,char_dim), kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.3)))
# # model.add(Dropout(0.5, noise_shape=None, seed=None))
# model.add(Dense(10, init='uniform', activation='relu'))
# model.add(Dropout(0.5, noise_shape=None, seed=None))
# model.add(Dense(5, init='uniform', activation='relu'))
# # model.add(Dropout(0.5, noise_shape=None, seed=None))
# model.add(Flatten())
# model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from keras.callbacks import ModelCheckpoint
# # Save the checkpoint in the /output folder
# filepath = "model/character-nn.hdf5"

# # Keep only a single checkpoint, the best over test accuracy.
# checkpoint = ModelCheckpoint(filepath,
#                             monitor='val_acc',
#                             verbose=1,
#                             save_best_only=True,
#                             mode='max',
#                             period=1)


# model.fit(X, y, epochs=5000, batch_size=10000,  verbose=1, validation_split=0.2, shuffle=True,callbacks=[checkpoint])

# model.save(filepath)
