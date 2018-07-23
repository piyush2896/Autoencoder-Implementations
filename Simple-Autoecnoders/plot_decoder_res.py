import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data import predict_input_fn, pred_labels
from model import model_func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist_autoecnoder = tf.estimator.Estimator(model_fn=model_func,
        params={'n_encode': 3, 'n_outs': 784}, model_dir='./autoencoder')

imgs = []

preds = mnist_autoecnoder.predict(input_fn=predict_input_fn)
for i in preds:
    imgs.append(i['decoder_out'])

imgs = np.asarray(imgs)

indices = np.random.randint(pred_labels.shape[0], size=100)

plt.figure(figsize=(10, 10))
for i, index in enumerate(indices):
    plt.subplot(10, 10, i+1)
    plt.imshow(imgs[index].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
