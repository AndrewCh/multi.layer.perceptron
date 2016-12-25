# Baseline MLP for MNIST dataset
import numpy
import skimage.io as io 
import os 
import platform
import getpass
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
from os.path import isfile, join

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
platform = platform.system()
currentUser = getpass.getuser()
currentDirectory = os.getcwd()

if platform is 'Windows':
	#path_image = 'C:\\Users\\' + currentUser
	path_image = currentDirectory 
else:	
	#path_image = '/user/' + currentUser
	path_image = currentDirectory 
fn = 'image.png'
img = io.imread(os.path.join(path_image, fn))

# prepare arrays
X_t = []
y_t = []
X_t.append(img)
y_t.append(3)

X_t = numpy.asarray(X_t)
y_t = numpy.asarray(y_t)
y_t = np_utils.to_categorical(y_t, 10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_t = X_t.reshape(X_t.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
X_t /= 255

print('X_train shape:', X_train.shape)
print ('X_t shape:', X_t.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_t.shape[0], 'test images')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print(y_test.shape[1], 'number of classes')

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
def build_model(model):
	# build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
    return model

def save_model(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
	
def load_model():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	if loaded_model:
		print("Loaded model")
	else:
		print("Model is not loaded correctly")
	return loaded_model

def print_class(scores):
	for index, score in numpy.ndenumerate(scores):
		number = index[1]
		print (number, "-", score)
	for index, score in numpy.ndenumerate(scores):
		if(score > 0.5):
			number = index[1]
			print ("\nNumber is: %d, probability is: %f" % (number, score))
	
model = baseline_model()
path = os.path.exists("model.json")
	
if not path:
	model = build_model(model)
	save_model(model)
	# Final evaluation of the model
	scores = model.predict(X_t)
	print("Probabilities for each class\n")
	print_class(scores)
else:
	# Final evaluation of the model
	loaded_model = load_model()
	if loaded_model is not None:
		loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		scores = loaded_model.predict(X_t)
		print("Probabilities for each class\n")
		print_class(scores)


