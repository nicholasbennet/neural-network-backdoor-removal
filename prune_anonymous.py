import tensorflow as tf
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot

#Congiguration for GPU
TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = TF_CONFIG_)

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

clean_data_filename = './data/clean_test_data.h5'

clean_test_filename = './data/clean_validation_data.h5'

poisoned_data_filename = './data/sunglasses_poisoned_data.h5'

model_filename = './models/anonymous_bd_net.h5'

model_weight_filename = './models/anonymous_bd_weights.h5'

x_test, y_test = data_loader(clean_test_filename)
x_test = data_preprocess(x_test)

x_train, y_train = data_loader(clean_data_filename)
x_train = data_preprocess(x_train)

x_pois, y_pois = data_loader(poisoned_data_filename)
x_pois = data_preprocess(x_pois)

bd_model = tf.keras.models.load_model(model_filename)

def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

base_model = tf.keras.models.load_model(model_filename)

base_model.load_weights(model_weight_filename) # optional but recommended.

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 5
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.68,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(base_model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

logdir = 'log'

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(x_train, y_train,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
class_accu = np.mean(np.equal(clean_label_p, y_test))*100
print('Classification accuracy for clean test datatset:', class_accu)

clean_label_pois_t = np.argmax(bd_model.predict(x_pois), axis=1)
class_accu_pois_t = np.mean(np.equal(clean_label_pois_t, y_pois))*100
print('Classification accuracy for poisoned dataset:', class_accu_pois_t)

clean_label_p_t = np.argmax(model_for_export.predict(x_test), axis=1)
class_accu_t = np.mean(np.equal(clean_label_p_t, y_test))*100
print('Classification accuracy for clean test datatset pruned model:', class_accu_t)

clean_label_pois = np.argmax(model_for_export.predict(x_pois), axis=1)
class_accu_pois = np.mean(np.equal(clean_label_pois, y_pois))*100
print('Classification accuracy for poisoned dataset pruned model:', class_accu_pois)

model_for_export.save("./models/pruned_anonymous_bd_net_temp.h5")