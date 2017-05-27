from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle
use_official_data = True


def generator(samples, batch_size=32):
    angle_adjust = [0, 0.2, -0.2]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # center, left, right
                for i in range(3):
                    if use_official_data:
                        name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    else:
                        name = './IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = cv2.resize(image, (0,0), fx=0.625, fy=0.625)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    angle = float(batch_sample[3]) + angle_adjust[i]
                    images.append(image)
                    angles.append(angle)
                    # flip image and angle
                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

if use_official_data:
    drivinglog = './data/driving_log.csv'
else:
    drivinglog = './driving_log.csv'
samples = []
with open(drivinglog) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
#         if float(line[3]) == 0:
#             continue
        samples.append(line)

BATCH_SIZE = 128
NB_EPOCH = 15

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

#### Training ####
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
# preprocessing
model.add(Cropping2D(cropping=((22,12), (0,0)), input_shape=(100,200,3)))
model.add(Lambda(lambda x: x / 255))

# conv -1
model.add(Convolution2D(nb_filter=8, nb_row=7, nb_col=7, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))
# conv 0
model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5, subsample=(1,1), border_mode='same'))
model.add(Activation('relu'))
# conv1
model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))
# conv2
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))
# conv3
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))
# conv4
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='valid'))
model.add(Activation('relu'))
# conv5
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Flatten())

# FC1
model.add(Dense(100))
model.add(Activation('relu'))
# FC2
model.add(Dense(50))
model.add(Activation('relu'))
# FC3
model.add(Dense(10))
model.add(Activation('relu'))
# Output
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=NB_EPOCH, verbose=1)
model.save('model.h5')
print('model saved.')
