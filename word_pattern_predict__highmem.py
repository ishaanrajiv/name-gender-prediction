import pandas as pd
import numpy as np
import re, string
import math
from keras.models import load_model
from utils import Utils

    
def main():
    util=Utils()
    try:
        encode_dict = util.wordPatternDict()
        gender = {0:'female',1:'male'}
        model = util.wordPatternModel
        while(1):
            name=input("\nEnter Name:")
            gender_arr = util.newWordPatternGenderPredict(name,encode_dict,model)
            print(gender[np.argmax(gender_arr)],np.max(gender_arr))
    except Exception as e:
        print("Aborting...\n",str(e))
    
if __name__ == "__main__":
    main()