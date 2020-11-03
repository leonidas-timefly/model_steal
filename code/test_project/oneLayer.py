from keras.layers import Dense, Dropout, Flatten
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from keras.layers import Activation
from keras.optimizers import RMSprop


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class_num = 20
input_dim = 224*224*3

x_train_all = np.load('trainData2.npy')
y_train_all = np.load('trainLabel2.npy')
x_test_all = np.load('testData2.npy')
y_test_all = np.load('testLabel2.npy')

#Radnomly select several target classes to re-train the model
train_set_ids = []
while len(train_set_ids) < class_num:
    new_id = np.random.randint(20) + 1    #randomly create int from[1,19]
    if new_id in train_set_ids:
        continue
    train_set_ids.append(new_id)

train_set = y_train_all==-1     #create a list with all False elements
test_set = y_test_all==-1
for id in train_set_ids:
    train_set = train_set | (y_train_all==id)
    test_set = test_set | (y_test_all==id)

x_train = x_train_all[train_set]
y_train = y_train_all[train_set]
x_test = x_test_all[test_set]
y_test = y_test_all[test_set]

x_train = tf.convert_to_tensor(x_train, dtype = tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype = tf.float32)
x_test = tf.convert_to_tensor(x_train, dtype = tf.float32)
y_test = tf.convert_to_tensor(x_train, dtype = tf.float32)

model = Sequential()

x = Dense(input_dim, name='fc6')(x_train)
output = Activation('softmax', name='fc6/softmax')(x)
custom_model = Model(model.input, output)

for layer in custom_model.layers:
    layer.trainable = True



custom_model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
custom_model.fit(x=x_train, y=y_train, epochs=20, batch_size=16)
result = custom_model.evaluate(x_test, y_test)

print(result)