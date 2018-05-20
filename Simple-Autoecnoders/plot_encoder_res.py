import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import predict_input_fn, pred_labels
from model import model_func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist_autoecnoder = tf.estimator.Estimator(model_fn=model_func,
        params={'n_encode': 3, 'n_outs': 784}, model_dir='./autoencoder')

imgs_3d = []

preds = mnist_autoecnoder.predict(input_fn=predict_input_fn)
for i in preds:
    imgs_3d.append(i['encoder_out'])

imgs_3d = np.asarray(imgs_3d)

colors = ['b', 'g', 'r', 'c', 'm',
          'y', 'k', '#bde123', '#123456', '#998855']

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
for i, color in enumerate(colors):
    ax.scatter(imgs_3d[pred_labels==i, 0], imgs_3d[pred_labels==i, 1], imgs_3d[pred_labels==i, 2], color=color)
plt.show()
