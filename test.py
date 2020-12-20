# visualize feature maps output from each block in the vgg model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import tensorflow as tf
import h5py
import numpy as np

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.InteractiveSession(config=config)

TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = TF_CONFIG_)

clean_data_filename = './data/clean_test_data.h5'

model_filename = './models/sunglasses_bd_net.h5'


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

x_test, y_test = data_loader(clean_data_filename)
x_test = data_preprocess(x_test)

bd_model = tf.keras.models.load_model(model_filename)

# load the model
model = bd_model
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()
feature_maps = model.predict(x_test)
length = 4
breadth = 5
# summarize feature map shapes
ix = 1
for _ in range(length):
	for _ in range(breadth):
		# specify subplot and turn of axis
		ax = pyplot.subplot(length, breadth, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()