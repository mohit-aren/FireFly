# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
import numpy
from numpy.random import RandomState as default_rng
#numpy.random.bit_generator = numpy.random._bit_generator
#from numpy.random import default_rng
import cv2
from keras import applications
from keras.models import Sequential
from keras.layers import  Convolution2D, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation
import random

img_width, img_height = 48, 48        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
model=applications.VGG16(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

#initialise top model
"""
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('vgg16_weights.h5', WEIGHTS_PATH_NO_TOP)

vgg_model.load_weights(weights_path)

# add the model on top of the convolutional base

model_final = Model(input= vgg_model.input, output= top_model(vgg_model.output))
"""
# Freeze first 15 layers
for layer in model.layers[:45]:
	layer.trainable = False
for layer in model.layers[45:]:
   layer.trainable = True


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(output_dim = 10, activation="softmax")(x) # 4-way softmax classifier at the end

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

onehot_encoder = OneHotEncoder(sparse=False)

y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


#model_final.fit(x_train,onehot_encoded,batch_size=64,epochs=1)

#model_final.save_weights('vgg16_1epoch.h5')
model_final.summary()
model_final.load_weights('vgg16_100epoch.h5')


#model1.load_weights('vgg16_1epoch.h5')
layer1_b = 128
layer2_b = 256
layer3_b = 512
layer4_b = 512

layer5_b = 1024
layer6_b = 1024

layer1_a = 128
layer2_a = 256
layer3_a = 512
layer4_a = 512

layer5_a = 1024
layer6_a = 1024

model1 = Sequential()
model1.add(Conv2D(input_shape=(48,48,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(10))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

y_test = y_test.reshape(len(y_test), 1)

onehot_encoded = onehot_encoder.fit_transform(y_test)

def enure_binary(x):
    y = []
    for indx in range(0, len(x)):
        if(x[indx] < 0.5):
            y.append(0)
        else:
            y.append(1)
            
    return y

####################### 1st convolution layer with 128 filters
print('1st convolution layer with 128 filters')
A = []
Acc = []

class FireflyAlgorithm:
    def __init__(self, pop_size=15, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def run(self, function, dim, lb, ub, max_evals):
        fireflies = [] #self.rng.uniform(lb, ub, (self.pop_size, dim))
        for idx1 in range(0, self.pop_size):
            vec = []
            for idx2 in range(0, dim):
                vec.append(random.uniform(0.001, 1.0))
            
            print(vec)
            fireflies.append(np.array(vec))
            
        fireflies = np.array(fireflies)
        intensity = np.apply_along_axis(function, 1, fireflies)
        print(intensity)
        best = np.max(intensity)
        print('best', best)
        best_firefly = fireflies[np.argmin(intensity, axis=-1)]
        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = ub - lb

        while evaluations <= max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] <= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.uniform(lb, ub, dim)-0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = function(fireflies[i])
                        evaluations += 1
                        if(intensity[i] > best):
                            best_firefly = fireflies[i]
                        best = max(intensity[i], best)
        return best_firefly
    
    
arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[4].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)

    for i in range(0,128):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        
    model_final.layers[4].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)  
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    print(score_trial)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=128, lb=0, ub=1, max_evals=100)
#print(best)
par1 = enure_binary(best)    
A1 = np.copy(par1)
new_num = np.sum(par1)
       
print(new_num)
layer1_a = new_num

####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[7].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    for i in range(0,256):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        
    model_final.layers[7].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=256, lb=0, ub=1, max_evals=100)

par1 = enure_binary(best)   
A2 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer2_a = new_num

####################### 1st convolution layer with 512 filters
print('1st convolution layer with 512 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[11].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    for i in range(0,512):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        
    model_final.layers[11].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=512, lb=0, ub=1, max_evals=100)

par1 = enure_binary(best)   

A3 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer3_a = new_num

####################### 2nd convolution layer with 512 filters
print('2nd convolution layer with 512 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[15].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    for i in range(0,512):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        
    model_final.layers[15].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=512, lb=0, ub=1, max_evals=100)

par1 = enure_binary(best)   

A4 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer4_a = new_num

####################### 1st dense layer with 1024 filters
print('1st dense layer with 1024 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[20].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    for i in range(0,1024):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, i] = 0
        
    model_final.layers[20].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=1024, lb=0, ub=1, max_evals=100)

par1 = enure_binary(best)   

A5 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer5_a = new_num


####################### 2nd dense layer with 1024 filters
print('2nd dense layer with 1024 filters')
A = []
Acc = []

arr = model_final.evaluate(x_test,onehot_encoded)
print(arr)

filters, biases = model_final.layers[22].get_weights()

def Fire_acc(x):
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    for i in range(0,1024):
        if(x[i] < 0.5):
            biases1[i] = 0
            filters1[:, i] = 0
        
    model_final.layers[22].set_weights([filters1, biases1])
    arr = model_final.evaluate(x_test,onehot_encoded)
    x = enure_binary(x)
    score_trial = 0.5*arr[1] + 0.5*len(x)/np.sum(x)
    return score_trial


FA = FireflyAlgorithm()
best = FA.run(function=Fire_acc, dim=1024, lb=0, ub=1, max_evals=100)

par1 = enure_binary(best)   

A6 = np.copy(par1)       
new_num = np.sum(par1)
       
print(new_num)
layer6_a = new_num


model1 = Sequential()
model1.add(Conv2D(input_shape=(48,48,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(10))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

layerr = model_final.layers[1].get_weights()
model1.layers[0].set_weights(layerr)

model = model_final
######################## 1st convolution layer with 128 filters
filters, biases = model.layers[4].get_weights()
filters1, biases1 = model1.layers[2].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 128, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(128):
    if(A1[j] == 1) :
        """
        for i1 in range (0,3):
            for j1 in range(0,3):
                filters1[:, :, index1][:,:,j][i1][j1] = filters[:, :, i][:,:,j][i1][j1]
        """
        filters1[:, :, :, index1] = filters[:, :, :, j]
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[2].set_weights([filters1, biases1])

######################## 1st convolution layer with 256 filters
filters, biases = model.layers[7].get_weights()
filters1, biases1 = model1.layers[4].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 256, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(256):
    if(A2[j] == 1) :
        index2 = 0
        for l in range(128):
            if(A1[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[4].set_weights([filters1, biases1])

######################## 1st convolution layer with 512 filters
filters, biases = model.layers[11].get_weights()
filters1, biases1 = model1.layers[6].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 512, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(512):
    if(A3[j] == 1) :
        index2 = 0
        for l in range(256):
            if(A2[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[6].set_weights([filters1, biases1])

######################## 2nd convolution layer with 512 filters
filters, biases = model.layers[15].get_weights()
filters1, biases1 = model1.layers[8].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 512, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(512):
    if(A4[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A3[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[8].set_weights([filters1, biases1])

######################## 1st dense layer with 1024 filters
filters, biases = model.layers[20].get_weights()
filters1, biases1 = model1.layers[11].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(1024):
    if(A5[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A4[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[11].set_weights([filters1, biases1])

######################## 2nd dense layer with 1024 filters
filters, biases = model.layers[22].get_weights()
filters1, biases1 = model1.layers[12].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(1024):
    if(A6[j] == 1) :
        index2 = 0
        for l in range(1024):
            if(A5[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)        
model1.layers[12].set_weights([filters1, biases1])


arr = model1.evaluate(x_test,onehot_encoded)
print(arr)

#y_train = y_train.reshape(len(y_train), 1)

onehot_encoded = onehot_encoder.fit_transform(y_train)


model1.fit(x_train,onehot_encoded,batch_size=64,epochs=100)
model1.summary()
model1.save_weights('VGG16_pruned_weights.h5')

#y_test = y_test.reshape(len(y_test), 1)

onehot_encoded = onehot_encoder.fit_transform(y_test)
arr = model1.evaluate(x_test,onehot_encoded)


print(arr)