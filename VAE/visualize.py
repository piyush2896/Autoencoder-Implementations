import matplotlib.pyplot as plt
from utils import get_filenames
import numpy as np
import cv2

def get_n_colors(n):
    colors = list(zip(np.random.randint(256, size=n),
                  np.random.randint(256, size=n),
                  np.random.randint(256, size=n)))
    colors = ['#{:02x}{:02x}{:02x}'.format(color[0],
                                           color[1],
                                           color[2]) for color in colors]
    return colors

def get_preds(pred_generator, key):
    preds = []

    for pred in pred_generator:
        preds.append(pred[key])
    return np.asarray(preds).astype('int')

def plot_decoder_out(pred_generator, n_imgs=5000, n_cols=10, n_rows=10):
    indices = np.random.randint(n_imgs, size=n_cols * n_rows)

    imgs_decoded = get_preds(pred_generator, 'decoder_out')

    plt.figure(figsize=(10, 10))
    for i, index in enumerate(indices):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(imgs_decoded[index])
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def plot_some_imgs(root):
    print('Displaying 100 images')

    filenames = get_filenames(root)
    indices = np.random.choice(len(filenames), size=100)
    plt.figure(figsize=(10, 10))
    for i, index in enumerate(indices):
        plt.subplot(10, 10, i+1)
        plt.imshow(cv2.cvtColor(cv2.imread(filenames[index]), cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()