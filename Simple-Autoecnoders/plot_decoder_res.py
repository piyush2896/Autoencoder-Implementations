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

indices = np.random.randint(pred_labels.shape[0], size=10)

plt.figure(figsize=(10, 10))
for i, index in enumerate(indices):
    plt.subplot(5, 2, i+1)
    plt.imshow(imgs[index].reshape(28, 28), cmap='binary')
    plt.axis('off')
    plt.title('Ground Truth:' + str(pred_labels[index]))
plt.show()
