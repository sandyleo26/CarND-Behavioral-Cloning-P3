from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # center, left, right
                cfile, lfile, rfile = ['./IMG/' + batch_sample[i].split('/')[-1] for i in range(3)]
                center_image = cv2.imread(cfile)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples = []
validation_samples = []
with open('./driving_log.csv') as csvfile:
    lines = list(csv.reader(csvfile))
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

BATCH_SIZE = 128
NB_EPOCH = 10
nb_filters_conv1 = 32
nb_filters_conv2 = 64
kernel_size = (5, 5)
pool_size = (2, 2)

model = Sequential()
# preprocessing
model.add(Lambda(lambda x: x / 255, input_shape=(160,320,3)))

# conv1
model.add(Convolution2D(nb_filters_conv1, kernel_size[0], kernel_size[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

# conv2
model.add(Convolution2D(nb_filters_conv2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())

# FC1
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC2
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
