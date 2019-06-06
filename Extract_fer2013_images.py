# importing necessary modules
import pandas as pd
import cv2
import numpy as np
import os
import shutil
 
def generate_modified_dataset():
	print("Generating Images ...")
	count = 0

	# reading csv file as pandas dataframe
	data = pd.read_csv(dataset_path)

	# converting numpy array into list
	pixels = data['pixels'].tolist()

	# obtaining list contain classes
	classes = sorted((data['emotion'].unique()).tolist())
	labels_column = data['emotion'].tolist()

	# creating classes subfolder
	for class_label in classes :
		os.mkdir(Extracted_Dataset_Location + '/' + str(class_label))

	# numpy array dimensions
	width, height = 48, 48
	
	# reading pixels and labels from csv to generate images
	for pixel_sequence,label in zip(pixels,labels_column):
		face = [int(pixel) for pixel in pixel_sequence.split(' ')]
		face = np.asarray(face).reshape(width, height)

		# resizing image
		face = cv2.resize(face.astype('uint8'),image_size)
		count = count + 1

		# setting image name
		image_name = str(count) + '.png'
	
		# Testing Purposes
		'''
		cv2.imshow('Image',face)
		cv2.waitKey(0)	
		'''
		# Saving the image
		cv2.imwrite(Extracted_Dataset_Location + '/' + str(label) + '/' + image_name,face)
	print("Image Generation done.")

	
dataset_path = 'fer2013/fer2013.csv'

# image dimensions
image_size=(200,200)

# save location of the modified dataset
Extracted_Dataset_Location = 'fer2013_extracted_Dataset'

# checking if extracted dataset already exists
if os.path.exists('fer2013_extracted_Dataset') == False :

	# creating extracted dataset folder
	os.mkdir('fer2013_extracted_Dataset') 
else:
	msg = int(input("Extracted Dataset Directory already present. Press 1 to use the existing directory or 2 to renew directory :"))
	if msg == 2 :

		# deleting extracted images
		shutil.rmtree('fer2013_extracted_Dataset')
		os.mkdir('fer2013_extracted_Dataset') 
		
generate_modified_dataset()
