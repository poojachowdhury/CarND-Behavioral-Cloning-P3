#importing all necessary libraries.
import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout ,  Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D

status=''
lines = []
cnt=0

# reading all the image paths skipping the first line as it contains the heading only.
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		if cnt==0:
			cnt=cnt+1
		else:
			lines.append(line)
			cnt=cnt+1

images=[]
measurements=[]

# storing the images and the steering angles in arrays after reading the images from the paths
for line in lines:
	source_path=line[0]
	filename=source_path.split('/')[-1]
	current_path='data/IMG/'+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	measurements.append(measurement)
	# include flipped images to increase the data for easy learning of car
	images.append(cv2.flip(image,1))		
	measurements.append(measurement*(-1.0))
	
	#include left images and their flipped versions
	source_path=line[1]
	filename=source_path.split('/')[-1]
	current_path='data/IMG/'+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	measurements.append(measurement)
	images.append(cv2.flip(image,1))
	measurements.append(measurement*(-1.0))

	#include right images and their flipped versions
	source_path=line[2]
	filename=source_path.split('/')[-1]
	current_path='data/IMG/'+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	measurements.append(measurement)

# convert the arrays to numpy arrays for further processing
X_train=np.array(images)
y_train=np.array(measurements)

# the NVIDIA architecture guided in udacity classroom
model = Sequential()
model.add(Cropping2D(cropping = ((65, 20), (2,2)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
# included drop out to reduce over fitting
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)

model.save('model.h5')