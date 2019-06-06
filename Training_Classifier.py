# Importing necessary modules
import os
import pandas as pd
import shutil
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping


if os.path.exists('saved_model'):
	shutil.rmtree('saved_model')

# Model save path
model_path="saved_model/model.ckpt"

# Prepairing and Splitting dataset for training and validation
def prep_dataset(dataset):
	print("ORIGINAL DATASET SAMPLE")
	print("--------------------------------")
	print(dataset.head(10))
	print("\n")

	print("ORIGINAL DATASET DIMENSIONS")
	print("--------------------------------")
	print(dataset.shape)
	print("\n")

	# storing features in x and converting it into numpy array
	x = dataset[dataset.columns[0:(dataset.shape[1]-1)]].values

	# storing labels in y
	y = dataset[dataset.columns[dataset.shape[1]-1]]

	# scaling features
	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	# number of unique classes
	n_classes = len(y.unique())

	# one-hot encoding the classes
	y=pd.get_dummies(y)
	
	# Shuffling the data
	x,y=shuffle(x,y,random_state=1)
	
	# Splitting dataset
	train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.20)
	
	return (train_x,test_x,train_y,test_y,n_classes)

# Computational Graph
def model_neural_network():
	model = Sequential()
	model.add(Dense(layer1_nodes, input_dim=n_cols, activation='relu'))
	model.add(Dense(layer2_nodes, activation='relu'))
	model.add(Dense(layer3_nodes, activation='relu'))	
	model.add(Dense(layer4_nodes, activation='relu'))
	model.add(Dense(n_classes, activation='softmax'))

	return model

# training model
def train_neural_network(model):
	optimizer = optimizers.Adam(lr = learning_rate)

	# callback for early stopping
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, verbose=1, mode='auto',restore_best_weights=True)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=epochs, batch_size = batch_size,validation_data=(x_test,y_test),callbacks = [early_stop])
	scores = model.evaluate(x_test, y_test)
	print("Accuracy = ",scores[1])	

# saving trained model
def save_model(model):
	if os.path.exists('Saved_Model') == False :
		os.mkdir('Saved_Model')
	pickle.dump(model, open('Saved_Model/Classifier_model.sav', 'wb'))

# Reading Dataset
dataset=pd.read_csv("features.csv")

x_train,x_test,y_train,y_test,n_classes=prep_dataset(dataset)

# number of types of features in the dataset
n_cols=x_train.shape[1]

# minibatch size
batch_size = 64

# number of epochs
epochs=45

# learning rate for optimizer
learning_rate = 0.0001

# setting nodes for deep layers
layer1_nodes = 512
layer2_nodes = 512
layer3_nodes = 512
layer4_nodes = 512

# calling functions
model = model_neural_network()
train_neural_network(model)
save_model(model)
