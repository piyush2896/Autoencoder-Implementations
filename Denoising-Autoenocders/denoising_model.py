import tensorflow as tf
import numpy as np

def dense(in_, units, name=None):
    return tf.layers.dense(inputs=in_,
                           units=units,
                           activation=tf.nn.leaky_relu,
                           name=name,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())

def conv2d(in_, filters, strides=2, padding='valid'):
    return tf.layers.conv2d(inputs=in_,
                            filters=filters,
                            kernel_size=3,
                            strides=strides,
                            padding=padding,
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconv2d(in_, filters, strides=2, padding='valid'):
    return tf.layers.conv2d_transpose(inputs=in_,
                                      filters=filters,
                                      kernel_size=3,
                                      strides=strides,
                                      padding=padding,
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

def vae_loss(encoder_out, decoder_out, labels):
    NAN_ELIMINATOR = 1e-7
    z_mean, z_stddev = encoder_out
    generation_loss = -tf.reduce_sum(labels * tf.log(NAN_ELIMINATOR + decoder_out) +
                                     (1 - labels) * tf.log(NAN_ELIMINATOR + (1 - decoder_out)),
                                     [1, 2, 3])
    divergence_loss = -0.5 * tf.reduce_sum(1 + tf.log(NAN_ELIMINATOR + tf.square(z_stddev)) -
                                           tf.square(z_mean) - tf.square(z_stddev), 1)

    loss = tf.reduce_mean(tf.add(generation_loss , divergence_loss))
    return loss

def dae_loss(decoder_out, labels):
    NAN_ELIMINATOR = 1e-7
    loss = -tf.reduce_sum(labels * tf.log(NAN_ELIMINATOR + decoder_out) +
                          (1 - labels) * tf.log(NAN_ELIMINATOR + (1 - decoder_out)),
                          axis=[1, 2, 3])
    return tf.reduce_mean(loss)

def encoder(features, latent_dims, is_vae=True):
    conv1 = conv2d(features['image'], 32, 1, 'same')
    conv2 = conv2d(conv1, 32, 1, 'same')
    norm1 = tf.layers.batch_normalization(conv2)

    conv3 = conv2d(norm1, 32, 1)
    conv4 = conv2d(conv3, 32, 1)
    norm2 = tf.layers.batch_normalization(conv4)

    conv5 = conv2d(norm2, 64, 1, 'same')
    conv6 = conv2d(conv5, 64, 1, 'same')
    norm3 = tf.layers.batch_normalization(conv6)

    conv7 = conv2d(norm3, 64, 1)
    conv8 = conv2d(conv7, 64, 1)
    norm4 = tf.layers.batch_normalization(conv8)

    conv9 = conv2d(norm4, 128, 1)
    conv10 = conv2d(conv9, 128, 1)
    norm5 = tf.layers.batch_normalization(conv10)

    flatten = tf.contrib.layers.flatten(norm5)

    dense1 = dense(flatten, latent_dims)

    if is_vae:
        z_mean = dense(dense1, latent_dims)
        z_stddev = dense(dense1, latent_dims)
        out = (z_mean, z_stddev)
    else:
        out = dense1

    return out, norm5.get_shape().as_list()[1:]

def decoder(encoder_out, output_channels, reshape_dim):
    dense1 = dense(encoder_out, np.product(reshape_dim))
    reshape = tf.reshape(dense1, [-1] + reshape_dim)

    deconv1 = deconv2d(reshape, 128, 1, 'same')
    deconv2 = deconv2d(deconv1, 128, 1)
    norm1 = tf.layers.batch_normalization(deconv2)

    deconv3 = deconv2d(norm1, 64, 1)
    deconv4 = deconv2d(deconv3, 64, 1)
    norm2 = tf.layers.batch_normalization(deconv4)

    deconv5 = deconv2d(norm2, 32, 1)
    deconv6 = deconv2d(deconv5, 32, 1)
    norm3 = tf.layers.batch_normalization(deconv6)

    deconv7 = tf.layers.conv2d_transpose(inputs=norm3,
                                         filters=output_channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='valid',
                                         activation=tf.nn.sigmoid)
    return deconv7

def model_fn(features, labels, mode, params, ):
    NAN_ELIMINATOR = 1e-7
    encoder_out, reshape_params = encoder(features,
                                          params['latent_dims'],
                                          is_vae=params['is_vae'])

    if params['is_vae']:
        z_mean, z_stddev = encoder_out
        sample = tf.random_normal(tf.shape(z_mean),
                              0, 0.1,
                              dtype='float')
        dist_out = z_mean + (z_stddev * sample)
    else:
        dist_out = encoder_out

    decoder_out = decoder(dist_out,
                          params['out_channels'],
                          reshape_params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'encoder_out': dist_out,
            'decoder_out': decoder_out * 255.
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:

        tf.summary.image('input/noised_img', features['image'] * 255.)
        tf.summary.image('input/label_img', labels * 255.)
        tf.summary.image('output', decoder_out * 255.)

        if params['is_vae']:
            loss = vae_loss(encoder_out, decoder_out, labels)
        else:
            loss = dae_loss(decoder_out, labels)

        global_step = tf.train.get_global_step()
        if params['lr_decay']:
            start_lr = params['lr']
            lr = tf.train.cosine_decay_restarts(start_lr,
                                                global_step,
                                                params['decay_steps'])
        else:
            lr = params['lr']
        tf.summary.scalar('lr', lr)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss,
                                      global_step=global_step)

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op)

    return spec
