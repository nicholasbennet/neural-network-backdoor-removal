#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:06:42 2020

@author: raskshithakoriraj
"""
#import torch
import h5py
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

tf.enable_eager_execution()
TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = TF_CONFIG_)

def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])
    
    return model


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'],dtype='float32')
    y_data = np.array(data['label'],dtype='float32')
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

@tf.function
def data_preprocess(x_data):
    return x_data/255

gen_x = keras.Input(shape=(55, 47, 3), name='input')
    # feature extraction
conv_1 = keras.layers.Conv2D(32, 9, padding='same', name='conv_1')(gen_x)
prelu_1 = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(conv_1)
res = res_block_gen(prelu_1, 3, 32, 1)
res = res_block_gen(res, 3, 32, 1)
res = res_block_gen(res, 3, 32, 1)
res = res_block_gen(res, 3, 32, 1)
#res_3 = res_block_gen(res_2, 3, 64, 1)
#res_4 = res_block_gen(res_3, 3, 64, 1)


conv_2 = keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(res)
batch_1 = keras.layers.BatchNormalization(momentum = 0.5)(conv_2)
model = keras.layers.add([prelu_1, batch_1])
        
conv_3 = keras.layers.Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same")(model)
act_1 = keras.layers.Activation('tanh')(conv_3)
       
model_gan = keras.models.Model(inputs = gen_x, outputs = act_1)
model_gan.save("gan_orig.h5")
#model_gan.summary()

clean_data_filename='data/clean_test_data.h5'
validation_data_filename = 'data/clean_validation_data.h5'
model_filename = 'models/anonymous_bd_net.h5'
x_train, y_train = data_loader(clean_data_filename)
x_train = data_preprocess(x_train)

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Prepare the training dataset.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0],seed=90).batch(batch_size)
bd_model_original = keras.models.load_model(model_filename)
bd_model_original.Training = False
epochs = 4
target_begin = 1

target_end = 10

for target in range(target_begin,target_end):
    K.clear_session()
    sess.close()
    sess = tf.compat.v1.Session(config = TF_CONFIG_)
    K.set_session(sess)

    #model_gan.set_weights(original_weights)
    model_gan = keras.models.load_model("gan_orig.h5")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:                
                logits = model_gan(x_batch_train, training=True)  # Logits for this minibatch                    
                #Scale between 0 and 1
                epsilon = 1e-12 
                logits = tf.div(
                    tf.subtract(logits, tf.reduce_min(logits)),
            tf.math.maximum( 
                    tf.subtract(tf.reduce_max(logits), tf.reduce_min(logits)),epsilon)
            )
                
                #Perturbation Loss
                l_pert = 0.01*tf.reduce_mean(tf.reduce_sum(tf.norm(logits,axis=(1,2)),axis=1))
                
                #l_pert = tf.clip_by_value(l_pert,0,100)
                y_predict = bd_model_original(
                    tf.clip_by_value(x_batch_train+logits,0,1), 
                    training=False)
                target_pred =  y_predict[:,target]
                if target == 0 :
                    k_th_pred = tf.math.reduce_max(y_predict[:,1:],axis=1)
                if target == 1282 :
                    k_th_pred = tf.math.reduce_max(y_predict[:,:1282],axis=1)
                else:
                    k_th_pred = tf.math.maximum(tf.math.reduce_max(y_predict[:,:target],axis=1),
                                                tf.math.reduce_max(y_predict[:,target+1:],axis=1))
                 
                l_adv = tf.math.reduce_mean(tf.math.maximum(k_th_pred- target_pred, 0))
                #print(l_pert.shape)
                a = 2.0
                if tf.is_nan(l_adv).numpy():
                    l_adv=0

                if tf.is_nan(l_pert).numpy():
                    l_pert=0

                if epoch > 0 and l_pert > l_adv:
                    a = 0.5

                #if epoch ==0:
                #    l_pert=0
                #    a=1

                loss_value =(a * l_adv) + l_pert
                #print(
                #    "loss_value:{} l_adv:{} l_pert:{} epoch:{}".format(loss_value, l_adv,l_pert,epoch)
                #)
                
            grads = tape.gradient(loss_value, model_gan.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_gan.trainable_weights))
            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step {}: target:{} l_adv:{} l_pert:{} loss:{} ".format(step,target,l_adv,l_pert,loss_value)
                )
                print("Seen so far: %s samples" % ((step + 1) * 64))

    model_gan.save("models/trigger_gen/trigger_gen_{}.h5".format(target))
    del model_gan
    del optimizer
     
