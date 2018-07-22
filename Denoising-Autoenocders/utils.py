from glob import glob
import cv2
import numpy as np
import os

def get_filenames(path, shuffle=True):
    filenames = glob(os.path.join(path, '*'))

    if shuffle:
        np.random.shuffle(filenames)
    return filenames

def load_img(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def train_dev_split(source, dev_path, split=0.1):
    filenames = get_filenames(source, shuffle=True)

    get_filename = lambda x: x.split('/')[-1]

    if not os.path.isdir(dev_path):
        os.makedirs(dev_path)

    print('Moving {} files'.format(int(len(filenames) * split)))
    split = len(filenames) - int(len(filenames) * split)
    files_to_move = filenames[split:]

    for file in files_to_move:
        dst = os.path.join(dev_path,
                           get_filename(file))
        os.rename(file, dst)
