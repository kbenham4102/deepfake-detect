import numpy as np
import pandas as pd
from utils import *
import os
import sys
import glob

def find_move_meta_json(path, label_path):
    
    json_file = glob.glob(path + '*.json')[0]
    
    cmd = 'mv ' + json_file + ' ' + label_path
    print('Executing ', cmd)
    os.system(cmd)
    fname = json_file.split('/')[-1]
    new_path = os.path.join(label_path, fname)
    return new_path
    

def sort_deepfake_train_examples(train_path, sorted_class_path, meta_path):
    # Load in a dataframe to parse out move protocol for train examples
    
    df = load_process_train_targets(meta_path, train_path)
    
    fakes_path = os.path.join(sorted_class_path, 'FAKE/')
    real_path = os.path.join(sorted_class_path, 'REAL/')
    
    # Check to see if path exists, if not, mkdir
    if not os.path.exists(sorted_class_path):
        os.mkdir(sorted_class_path)
    if not os.path.exists(fakes_path):
        os.mkdir(fakes_path)
        print('created path ', fakes_path)
    if not os.path.exists(real_path):
        os.mkdir(real_path)
        print('created path ', real_path)

    n = len(df)
    print("Num train videos ", n)
    for i in range(n):
        fpath = df.iloc[i].filepath
        cls = df.iloc[i].target_class
        
        if cls == 1.0:
            # if these are 1.0 they are a fake
            os.system('mv ' + fpath + ' ' + fakes_path)
            
        else:
            os.system('mv ' + fpath + ' ' + real_path)
        
        if (i+1)%100 == 0:
            print("Successfully sorted {} examples".format(i+1))