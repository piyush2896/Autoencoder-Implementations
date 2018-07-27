import tensorflow as tf

def conv2d(in_, filters, k_size, strides=2, padding='same'):
    return tf.layers.conv2d(inputs=in_,
                            filters=filters,
                            kernel_size=k_size,
                            strides=strides,
                            padding=padding,
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconv2d(in_, filters, k_size, strides=2, padding='same'):
    return tf.layers.conv2d_transpose(inputs=in_,
                                      filters=filters,
                                      kernel_size=k_size,
                                      strides=strides,
                                      padding=padding,
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

def merge(X1, X2, channels=None, act=tf.nn.leaky_relu):
    with tf.name_scope('merger'):
        if channels is not None:
            re_channel = conv2d(X1, channels, 1, 1)
            return act(tf.add(re_channel, X2))
        return act(tf.add(X1, X2))

def model_fn(features, labels, mode, params):
    conv_balancer = conv2d(features['image'], params['out_channels'], 1, 1)

    conv1 = conv2d(conv_balancer, 32, 3, 2)
    conv2 = conv2d(conv1, 32, 3, 2)

    conv3 = conv2d(conv2, 64, 3, 2)
    conv4 = conv2d(conv3, 64, 3, 2)

    conv5 = conv2d(conv4, 128, 3, 2)

    deconv1 = deconv2d(conv5, 128, 3, 2)
    # skip connection - connect input of conv5 to output of deconv1
    # make sure to take care of those extra channels using 1x1 convs
    merge41 = merge(conv4, deconv1, 128)

    deconv2 = deconv2d(merge41, 64, 3, 2)
    merge32 = merge(conv3, deconv2)

    deconv3 = deconv2d(merge32, 64, 3, 2)
    merge23 = merge(conv2, deconv3, 64)

    deconv4 = deconv2d(merge23, 32, 3, 2)
    merge14 = merge(conv1, deconv4)

    deconv6 = deconv2d(merge14, params['out_channels'], 3, 2)
    merge_balancer6 = merge(conv_balancer, deconv6, act=tf.nn.sigmoid)

    out = tf.multiply(merge_balancer6, tf.constant(255.))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'decoder_out': out
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        
        tf.summary.image('input', labels * 255.)
        tf.summary.image('out', out)
        loss = tf.losses.mean_squared_error(labels, merge_balancer6)
        optimizer = tf.train.AdamOptimizer(params['lr'])
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op)

    return spec
