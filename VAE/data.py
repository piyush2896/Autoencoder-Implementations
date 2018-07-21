import tensorflow as tf
from utils import get_train_data, get_test_data, get_filenames

def input_fn_helper(filenames,
                    img_size,
                    is_test=False):
    def load(filename):
        bytes = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(bytes, channels=3)
        img_resized = tf.image.resize_images(img_decoded, img_size)
        img_normalized = tf.divide(tf.cast(img_resized, 'float'), tf.constant(255.))
        img_flipped = tf.image.random_flip_left_right(img_normalized)
        
        if is_test:
            return img_flipped
        return img_flipped, img_flipped

    dataset = tf.data.Dataset.from_tensor_slices((filenames, ))
    dataset = dataset.map(load)
    return dataset

def train_input_fn(root,
                   img_size,
                   batch_size,
                   buffer_size):
    filenames = get_filenames(root,
                              shuffle=True,
                              is_test=True)

    dataset = input_fn_helper(filenames, img_size)

    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.repeat()

    images_x, images_y = dataset.make_one_shot_iterator().get_next()

    features_dic = {'image': images_x}
    return features_dic, images_y

def eval_input_fn(root,
                  img_size,
                  batch_size,
                  buffer_size):
    filenames = get_filenames(root,
                              shuffle=False,
                              is_test=True)

    dataset = input_fn_helper(filenames, img_size)
    dataset = dataset.prefetch(buffer_size).batch(batch_size)
    dataset = dataset.repeat(1)

    images_x, images_y = dataset.make_one_shot_iterator().get_next()

    features_dic = {'image': images_x}
    return features_dic, images_y

def predict_input_fn(root,
                     img_size,
                     batch_size,
                     buffer_size):
    filenames = get_filenames(root,
                              shuffle=False,
                              is_test=True)
    dataset = input_fn_helper(filenames, img_size, is_test=True)
    dataset = dataset.prefetch(buffer_size).batch(batch_size)
    dataset = dataset.repeat(1)

    images_x = dataset.make_one_shot_iterator().get_next()

    features_dic = {'image': images_x}
    return features_dic
