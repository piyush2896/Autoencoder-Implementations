import numpy as np
import cv2
from tqdm import tqdm
import os
from glob import glob
import random
import matplotlib.pyplot as plt

def read_img(path, size=(64, 64)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size)

def get_train_data(path):
    raw_labels = os.listdir(path)
    imgs = []
    labels = []
    for i, raw_label in enumerate(tqdm(raw_labels)):
        img_names = os.listdir(os.path.join(path, raw_label))
        for img_name in img_names:
            if img_name in ['.DS_Store', '._.DS_Store']:
                continue
            imgs.append(read_img(os.path.join(path, raw_label, img_name)))
        labels = labels + [i] * len(img_names)
    return np.asarray(imgs, dtype=np.float32), np.asarray(labels), raw_labels

def get_test_data(path):
    img_names = os.listdir(path)
    imgs = []
    labels = []
    for img_name in tqdm(img_names):
        if img_name in ['.DS_Store', '._.DS_Store']:
            continue
        imgs.append(read_img(os.path.join(path, img_name)))
        labels.append('_'.join(img_name.split('_')[:-1]))
    return np.asarray(imgs, dtype=np.float32), labels

def get_filenames(root, shuffle=False, is_test=False):
    if is_test:
        path = os.path.join(root, '*')
    else:
        path = os.path.join(root, '*')

    filenames = glob(path, recursive=True)

    if shuffle:
        random.shuffle(filenames)
    return filenames

def train_dev_split(source, dev_path, split=0.1):
    filenames = get_filenames(source, shuffle=True, is_test=True)

    labels = glob(os.path.join(source, '*'))

    get_filename = lambda x: x.split('/')[-1]

    if not os.path.isdir(dev_path):
        os.makedirs(dev_path)

    print('Moving {} files'.format(int(len(filenames) * split)))
    split = len(filenames) - int(len(filenames) * split)
    files_to_move = filenames[split:]

    for file in files_to_move:
        dst = os.path.join(dev_path,
                           #get_label(file),
                           get_filename(file))
        os.rename(file, dst)

def get_labels(root):
    get_label = lambda x: ' '.join(x.split('\\')[1].split('_')[:-1])
    preds_labels = list(map(get_label, glob(os.path.join(root, '*'))))
    return preds_labels
