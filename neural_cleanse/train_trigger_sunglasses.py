from tensorflow import keras
import h5py
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

clean_data_filename='data/clean_test_data.h5'
poisoned_data_filename='data/sunglasses_poisoned_data.h5'
trigger_filename='triggers/sunglasses.h5'
validation_data_filename='data/clean_validation_data.h5'
model_filename = 'models/sunglasses_bd_net.h5'
pruned_model_filename = 'models/pruned_sunglasses_bd_net_temp.h5'


bd_model_original = keras.models.load_model(model_filename)

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test_clean, y_test_clean = data_loader(clean_data_filename)
    x_test_clean = data_preprocess(x_test_clean)
    x_trig, y_trig = data_loader(trigger_filename)
    x_trig = x_trig.transpose((0,3,1,2))
    x_trig = data_preprocess(x_trig)
    x_test_poisoned, y_test_poisoned = data_loader(poisoned_data_filename)
    x_test_poisoned = data_preprocess(x_test_poisoned)
    
    x_test = np.vstack((x_test_clean, x_trig))
    y_test = np.hstack((y_test_clean, np.zeros(y_trig.shape)+np.max(y_test_clean)+1))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(100)
    
    bd_model = keras.models.load_model(pruned_model_filename)
    new_output = keras.layers.Dense(1284, activation='softmax',name='output')(bd_model.layers[-2].output)
    bd_model = keras.Model(inputs=bd_model.inputs, outputs=new_output)
    bd_model.compile(
        optimizer=bd_model_original.optimizer,
        loss=bd_model_original.loss,
        metrics=['accuracy']
    )
    bd_model.fit(train_dataset,epochs=10)
    
    x_val, y_val = data_loader(validation_data_filename)
    x_val = data_preprocess(x_val)
    
    
    clean_label_p = np.argmax(bd_model_original.predict(x_test_clean), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test_clean))*100
    print('Classification accuracy for Clean test data:', class_accu)
    
    clean_label_p = np.argmax(bd_model.predict(x_test_clean), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test_clean))*100
    print('Classification accuracy for Clean test data retrained:', class_accu)
    
    clean_label_p = np.argmax(bd_model_original.predict(x_test_poisoned), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test_poisoned))*100
    print('Classification accuracy for Clean poisoned data:', class_accu)
    
    clean_label_p = np.argmax(bd_model.predict(x_test_poisoned), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test_poisoned))*100
    print('Classification accuracy for Clean poisoned data retrained:', class_accu)
    
    clean_label_p = np.argmax(bd_model_original.predict(x_val), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_val))*100
    print('Classification accuracy for validation data:', class_accu)
    
    clean_label_p = np.argmax(bd_model.predict(x_val), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_val))*100
    print('Classification accuracy for validation data retrained:', class_accu)
    
    bd_model.save("./models/retrain_with_trig_sunglasses_bd_net.h5")

if __name__ == '__main__':
    main()
    session.close()