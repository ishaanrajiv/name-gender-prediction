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
from keras.layers import Flatten

### Config Parameters
min_name_length = 3 # Min length of name
max_name_length = 20

### Read Train Dataset - Name,Gender header
df = pd.read_csv("dataset/train.csv")

### Remove names with less than 2 characters
df = df[df['Name'].map(str).map(len) >= min_name_length][df['Name'].map(str).map(len) <= max_name_length]

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

def cleanName(dataframe):
    dataframe = dataframe['Name']
    dataframe = dataframe.apply(lambda x: encodeNameString(x))
    return pd.DataFrame(dataframe)


df_X = cleanName(df)

for i in range(0,8):
    df_X[i]=df_X['Name'].apply(lambda x: x[i])


# from functools import reduce
# dfs = [pd.get_dummies(df_X['Name'].apply(lambda x: x[i])) for i in range(0,max_name_length)]
# df_X = pd.concat(dfs, axis=1)

df_X = df_X.drop(columns=['Name'])
X = df_X.values

X =np.array([(((x[:,None] & (1 << np.arange(char_dim)))) > 0).astype(int) for x in df_X.values])

df_Y = pd.get_dummies(df['Gender'])
y = df_Y.values

input_dim = len(X[0])
output_dim = len(y[0])

model = Sequential()
model.add(Dense(input_dim+1, input_shape=(input_dim,char_dim), kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.3)))
# model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(5, init='uniform', activation='relu'))
# model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Flatten())
model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
# Save the checkpoint in the /output folder
filepath = "model/word-pattern-nn.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max',
                            period=1)


model.fit(X, y, epochs=5000000, batch_size=15000,  verbose=1, validation_split=0.2, shuffle=True,callbacks=[checkpoint])

model.save(filepath)
