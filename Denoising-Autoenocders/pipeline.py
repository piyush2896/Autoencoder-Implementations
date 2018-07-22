import tensorflow as tf

def corrupt(x):
    return tf.multiply(x, tf.random_normal(shape=tf.shape(x),
                mean=0, stddev=2, dtype=tf.float32))

def shuffle_repeat_applier(dataset, buffer_size, shuffle, repeat):
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    else:
        dataset = dataset.prefetch(buffer_size)

    if repeat:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(1)

    return dataset

def input_fn(filenames,
             img_size,
             noising_ratio=0.4,
             batch_size=32,
             buffer_size=2000,
             shuffle=True,
             repeat=True):

    def load_and_add_noise(filename):
        bytes = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(bytes, channels=3)
        img_resized = tf.image.resize_images(img_decoded, img_size)
        img_normalized = tf.divide(tf.cast(img_resized, 'float'), tf.constant(255.))

        img_corrupted_part = tf.multiply(corrupt(img_normalized), 
                                         tf.constant(noising_ratio))
        img_safe_part = tf.multiply(img_normalized, tf.constant(1-noising_ratio))
        img_noised = tf.add(img_corrupted_part, img_safe_part)
        img_noised = tf.divide(img_noised, tf.reduce_max(img_noised))

        return img_noised, img_normalized

    dataset = tf.data.Dataset.from_tensor_slices((filenames, ))
    dataset = dataset.map(load_and_add_noise)

    dataset = shuffle_repeat_applier(dataset, buffer_size, shuffle, repeat)

    dataset = dataset.batch(batch_size)

    images_x, images_y = dataset.make_one_shot_iterator().get_next()

    features_dic = {'image': images_x}
    return features_dic, images_y
