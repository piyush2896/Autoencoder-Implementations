import tensorflow as tf
from utils import get_data

(train_data, train_labels), (eval_data, eval_labels) = get_data()

train_input_func = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_data,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_data,
    num_epochs=1,
    shuffle=False)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_data},
    num_epochs=1,
    shuffle=False)

pred_labels = train_labels