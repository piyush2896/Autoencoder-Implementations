from data import train_input_func, eval_input_fn, predict_input_fn
from model import model_func
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist_autoecnoder = tf.estimator.Estimator(model_fn=model_func,
        params={'n_encode': 3, 'n_outs': 784}, model_dir='./autoencoder')

mnist_autoecnoder.train(input_fn=train_input_func,
                        steps=2000)

eval_res = mnist_autoecnoder.evaluate(input_fn=eval_input_fn)

print(eval_res)

