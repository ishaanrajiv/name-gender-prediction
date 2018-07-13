import pandas as pd
import numpy as np
import re, string
import math
from keras.models import load_model
from utils import Utils

    
def main():
    try:
        encode_dict = Utils().characterDict()
        gender = {0:'female',1:'male'}
        model = Utils().characterModel
        while(1):
            name=input("\nEnter Name:")
            gender_arr = Utils().characterGenderPredict(name,encode_dict,model)
            print(gender[np.argmax(gender_arr)],np.max(gender_arr))
    except Exception as e:
        print("Aborting...\n",str(e))
    
if __name__ == "__main__":
    main()
