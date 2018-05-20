import tensorflow as tf

def get_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    eval_imgs = mnist.test.images
    eval_labels = mnist.test.labels
    return (train_imgs, train_labels), (eval_imgs, eval_labels)
