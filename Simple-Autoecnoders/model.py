import tensorflow as tf

def dense(in_, units, name=None):
    return tf.layers.dense(inputs=in_,
                           units=units,
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.contrib.layers.xavier_initializer,
                           name=name)

def encoder(features, n_encode):
    dense1 = dense(features['x'], 512, name='en_dense1')
    dense2 = dense(dense1, 256, name='en_dense2')
    dense3 = dense(dense2, 120, name='en_dense3')
    dense4 = dense(dense3, n_encode, name='encoder_out')
    return dense4

def decoder(encoder_out, n_outs):
    dense1 = dense(encoder_out, 120, name='de_dense1')
    dense2 = dense(dense1, 256, name='de_dense2')
    dense3 = dense(dense2, 512, name='de_dense3')
    dense4 = dense(dense3, n_outs, name='decoder_out')
    return dense4

def model_func(features, labels, mode, params):
    encoder_out = encoder(features, params['n_encode'])
    decoder_out = decoder(encoder_out, params['n_outs'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'encoder_out': encoder_out,
            'decoder_out': decoder_out
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        loss = tf.losses.mean_squared_error(labels, decoder_out)
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op)

    return spec
