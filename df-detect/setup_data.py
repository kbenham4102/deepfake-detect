import numpy as np
import pandas as pd
from utils import load_process_train_targets
import os
import sys
import glob
import argparse
from sklearn.model_selection import train_test_split
import cv2


# TODO Set this up to updaste based on new zip file download
# 1. Download all zip files from kaggle page
# 2. Run this script from the command line to extract, remove zip
# and then move video files to a structured directory basedon their real or fake label

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", help="Folder for downloaded zipped video files")
    # parser.add_argument("-extract-bucket", help="location to find json meta data file and images after extraction")
    parser.add_argument("-lp", help="label path to store json labels")
    parser.add_argument("-im-sort-path", help="path to store sorted images")
    parser.add_argument("--train-val-split",type=float, help="percent to hold out as validation set", default=0.015)
    parser.add_argument("--json-to-update", type=str, help="if set, loads an existing json to append more data to")

    args = parser.parse_args()

    return args

def extract_zip_and_remove(download_folder, zip_file):

    # Extract where the .zip file lives
    zip_loc = os.path.join(download_folder, zip_file)
    print(zip_loc)

    # Unzip the file
    cmd = "unzip -q {} -d {}".format(zip_loc, download_folder)
    
    os.system(cmd)

    # Remove the zip file to free memory
    

    unzipped_fold = zip_loc.split('.')[:-1][0]
    if os.path.exists(unzipped_fold):
        unzipped_folder = unzipped_fold
    else:
        if 'part_00' in zip_file:
            # Handle the case where unzip truncates a zero in the output folder
            unzipped_folder = zip_loc.split('.')[:-1][0][:-1]
        elif 'part_0' in zip_file:
            pre = zip_loc.split('.')[:-1][0][:-2]
            last_char = zip_loc.split('.')[:-1][0][-1]
            unzipped_folder = pre + last_char
            assert os.path.exists(unzipped_folder)
    
    
    os.system("rm " + zip_loc)
        

    return unzipped_folder


def find_move_meta_json(path, label_path):
    
    json_file = glob.glob(path + '*.json')[0]
    
    cmd = 'mv ' + json_file + ' ' + label_path
    print('Executing ', cmd)
    os.system(cmd)
    fname = json_file.split('/')[-1]
    new_path = os.path.join(label_path, fname)
    return new_path


def move_vids(df, fakes_path, real_path):
    n = len(df)
    print("Num videos ", n)
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
    print("Successfully sorted {} examples".format(i+1))

def sort_deepfake_train_examples(train_path, sorted_class_path, meta_path, test_split = 0.2):
    # Load in a dataframe to parse out move protocol for train examples
    
    df = load_process_train_targets(meta_path, train_path)

    df_train, df_val = train_test_split(df, test_size=test_split)

    print("Train set len ", len(df_train))
    print("Val set len ", len(df_val))
    
    train_fakes_path = os.path.join(sorted_class_path, 'train', 'FAKE/')
    train_real_path = os.path.join(sorted_class_path, 'train', 'REAL/')
    val_fakes_path = os.path.join(sorted_class_path, 'val', 'FAKE/')
    val_real_path = os.path.join(sorted_class_path, 'val', 'REAL/')

    # Check to see if path exists, if not, mkdirs
    if not os.path.exists(sorted_class_path):
        # Make the head directory
        os.mkdir(sorted_class_path)
        os.mkdir(os.path.join(sorted_class_path, 'train'))
        os.mkdir(os.path.join(sorted_class_path, 'val'))
        os.mkdir(train_fakes_path)
        os.mkdir(train_real_path)
        os.mkdir(val_fakes_path)
        os.mkdir(val_real_path)

        print("PATHS CREATED:\n" + 
              sorted_class_path + '\n' +
              train_fakes_path + '\n' +
              train_real_path +'\n' +
              val_fakes_path + '\n' +
              val_real_path + '\n'
              )

    move_vids(df_train, train_fakes_path, train_real_path)
    move_vids(df_val, val_fakes_path, val_real_path)


def re_sort(fraction, mode = 'val'):
    """
    adhoc function to resort train val split, not really needed with new dataloader
    
    Arguments:
        fraction {[type]} -- [description]
    
    Keyword Arguments:
        mode {str} -- [description] (default: {'val'})
    """

    if mode == 'val':
        l = glob.glob('../data/source/train_val_sort/val/*/*.mp4')
    elif mode == 'train':
        l = glob.glob('../data/source/train_val_sort/train/*/*.mp4')
    mod_no = int(1/fraction)
    i = 0
    for file in l:    
        parts = file.split('/')
        label = parts[-2]
        if i%mod_no ==0:
            if label == 'REAL':
                cmd = f'mv {file} data/source/train_val_sort/{mode}/REAL/'
                os.system(cmd)
            elif label == 'FAKE':
                cmd = f'mv {file} data/source/train_val_sort/{mode}/FAKE/'
                os.system(cmd)
        i += 1

def get_nframes_test(path):
    capture = cv2.VideoCapture(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count
    
def test_frames_clean(df):

    i = 0
    arr = df.to_numpy()
    for r,f in arr:
        fs = get_nframes_test(f)
        rs = get_nframes_test(r)
        i+=1
        if i%10000 == 0:
            print('vids tested: ', i+1)
        
        if fs != rs:
            print(f'found bad pair: \n{r}\n{f}')
            ind = df[df.real == r].index
            df.drop(ind, inplace=True)
    return df

            


def etl_save_pairs(json_path, out_path, 
                   data_head = '/home/kevin/data/deepfakes_data/source/train_val_sort/*/*/', check_dims = False):


    #   
    df = pd.read_json(json_path)
    df_fakes = df[df['label'] == 'FAKE']
    fake_names = df_fakes['index']
    df_fakes['fake_name'] = fake_names

    df_out = pd.DataFrame()

    fakes_paths = []
    reals_paths = []
    i = 0
    for r,f in df_fakes[["original","fake_name"]].to_numpy():

        r_path = glob.glob(data_head + r)
        f_path = glob.glob(data_head + f)

        if len(r_path) != 0 and len(f_path) != 0:
            reals_paths.append(r_path[0])
            fakes_paths.append(f_path[0])
        i +=1
        if i%100 == 0:
            print('files matched: ', len(reals_paths))
    
    df_out['fake'] = fakes_paths
    df_out['real'] = reals_paths

    del df_fakes, df
    if check_dims:
        df_out_final = test_frames_clean(df_out)
        print('total files cleaned, matched: ', len(df_out_final))
    else:
        df_out.to_csv(os.path.join(out_path, 'fake_to_real_mapping.csv'))
        print('total files cleaned, matched: ', len(df_out))

def extract_and_move_data():
    """
    Steps:
    1. Make sure all zip files are in the `-d` arg directory. 
    2. Specify where to store the resulting json dictionary for labels and video pairs
    3. specify a head directory to store the data
    example usage:

    python3 setup_data.py -d ~/Downloads/deepfakes -lp ~/data/deepfakes_data/source/labels/ --im-sort-path ~/data/deepfakes_data/source/train_val_sort/
    """
    # Get the filepath args
    args = parse_args()

    download_folder = args.d



    zips = glob.glob(os.path.join(download_folder, '*.zip'))
    print(zips)

    if len(zips) > 0:
        i = 0
        for z in zips:
            print("EXTRACTING AND REMOVING ZIPS")

            unzipped_folder = extract_zip_and_remove(download_folder, z)

            # Move the videos to the deepfakes directory, remove the empty dir
            os.system(f"mv {unzipped_folder}/*.mp4 {download_folder}")
            
            print(f"Extracted {i+1} zips")
            i+=1

    
    json_file_list = glob.glob(os.path.join(download_folder, '*/*.json'))
    print(json_file_list)
    assert len(json_file_list) > 0

    i = 0
    for jf in json_file_list:
        print("appending ", jf)
        if i == 0:
            if args.json_to_update:
                # TODO make this work with new jsons loaded
                df_master = pd.read_json(args.json_to_update) # Should be formatted correctly
            else:
                df_master = pd.read_json(jf, orient='index').reset_index()
        else:
            df_temp = pd.read_json(jf, orient='index').reset_index()
            df_master = df_master.append(df_temp)
        i+=1

    df_master = df_master.reset_index().drop('level_0', axis=1)
    # Save the json to the source label directory
    json_loc = os.path.join(args.lp, "master_meta.json")
    df_master.to_json(json_loc)
    print("saved new json file ", json_loc)

    sort_deepfake_train_examples(download_folder, args.im_sort_path, json_loc, test_split=args.train_val_split)
    return json_loc, args.im_sort_path

if __name__== "__main__":

    json_loc, data_head = extract_and_move_data()
    out_list = json_loc.split('/')[:-1]
    out_path = os.path.join(*out_list)

    # Run a function to create a master list for matched originals and fakes
    etl_save_pairs(json_loc, out_path, data_head=os.path.expanduser(data_head))





