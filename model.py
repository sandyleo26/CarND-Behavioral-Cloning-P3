from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import csv
import cv2
import numpy as np

BATCH_SIZE = 128
NB_EPOCH = 10

nb_filters_conv1 = 32
nb_filters_conv2 = 64

kernel_size = (5, 5)
pool_size = (2, 2)

images = []
angles = []
with open('./driving_log.csv') as csvfile:
    lines = list(csv.reader(csvfile))
    for line in lines:
        tokens = line
        cfile, lfile, rfile = ['./IMG/' + tokens[i].split('/')[-1] for i in range(3)]
        angle = float(tokens[3])

        image = cv2.imread(cfile)
        images.append(image)
        angles.append(angle)
        
X_train = np.array(images)
print(X_train.shape)
y_train = np.array(angles)
print(y_train.shape)

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
model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, verbose=1, validation_split=0.2)
model.save('model.h5')
