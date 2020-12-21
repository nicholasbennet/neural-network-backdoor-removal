import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
tf.compat.v1.disable_eager_execution()
def dump_image(x, filename, format):
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)
    return

#tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction)

def fix_gpu_memory(mem_fraction=1):
    
    with tf.compat.v1.Session() as sess:
        from tensorflow.compat.v1.keras import backend as K
        #tf.compat.v1.disable_early_execution()

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
        tf_config.allow_soft_placement = True
        init_op = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(config=tf_config)
        sess.run(init_op)
        K.set_session(sess)
        return sess


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
