from code.test_project import VGGFace
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class_num = 20
input_dim = 50*50*3

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
#calculate the num of random int from x_train_all/x_test_all/y_train_all/y_test_all

#change the range of target class labels to (1, n)
for i in range(0,y_train.shape[0]):
    y_train[i] = train_set_ids.index(y_train[i]) + 1
for i in range(0, y_test.shape[0]):
    y_test[i] = train_set_ids.index(y_test[i]) + 1

temp = np.zeros((y_train.shape[0], int(np.max(y_train))))
temp[np.arange(y_train.shape[0]), y_train.astype(int) - 1] = 1
y_train = temp

temp = np.zeros((y_test.shape[0], int(np.max(y_test))))
temp[np.arange(y_test.shape[0]), y_test.astype(int) - 1] = 1
y_test = temp


print("training begins")

model = VGGFace(model='VGG16', pooling='max')
last_layer = model.get_layer('conv1_1').output

x = Dense(class_num, name='fc6')(last_layer)
output = Activation('softmax', name='fc6/softmax')(last_layer)
custom_model = Model(model.input, output)

for layer in custom_model.layers:
    layer.trainable = True



custom_model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
custom_model.fit(x=x_train, y=y_train, epochs=20, batch_size=16)
result = custom_model.evaluate(x_test, y_test)

print(result)