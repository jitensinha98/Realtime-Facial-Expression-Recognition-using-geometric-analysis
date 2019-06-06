# importing necessary modules
import cv2
import os
import numpy as np
from tqdm import tqdm
import dlib
from imutils import face_utils
import csv
import pandas as pd
from scipy.spatial import distance

# Generating Bounding Boxes coordinates for plotting
def compute_bounding_boxes_dimensions(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

# saving features to a csv file
def save_features_to_csv(euclidean_distance_list):

	# saving features ins a 'features.csv' file
	with open('features.csv', 'a' , newline = '') as myfile:
		if len(euclidean_distance_list) == 4625 :
			wr = csv.writer(myfile)
			wr.writerow(euclidean_distance_list)

# extracting features from images and saving them
def get_and_save_features(image,names):
	coordinates_list = []
	euclidean_distance_list = []

	# converting into gray scale image . We have done this in 'Data_Preparation.py' too and we are it here just to avoid any unexpected error.
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# detecting rectangle dimensions for faces
	rects = detector(gray_image,0)
	if len(rects) > 0:
		for rect in rects:
			shape = predictor(gray_image, rect)
			shape = face_utils.shape_to_np(shape)
			for elements in shape.tolist():
				coordinates_list.append(elements)
				
			for i in range(0,len(coordinates_list)):
				for item in coordinates_list:
					# calculating euclidean distances between each landmark point i.e. 68x68 = 4624 features
					euclidean_distance = distance.euclidean(coordinates_list[i], item)
					euclidean_distance_list.append(euclidean_distance)
			
			# appending class labels		
			euclidean_distance_list.append(names)	

			save_features_to_csv(euclidean_distance_list)

			# Testing Purposes
			'''
			(x,y,w,h) = compute_bounding_boxes_dimensions(rect)
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			for (x, y) in shape:
            			cv2.circle(image, (x, y), 2, (0,0, 255), -1)
			cv2.imshow("Output", image)
			cv2.waitKey(0)
			'''
# Reads images 
def prepare_features_csv():
	print("------Generating Face Landmark coordinates CSV------")
	print(" ")
	if os.path.exists('features.csv') :
		os.remove('features.csv')
	for names in os.listdir(dataset_path):
		img_folder = os.path.join(dataset_path,names)

		print(" ")
		print("Processing Label ", names ," Images ")
		print(" ")
		
		for image in tqdm(os.listdir(img_folder)):
			imgr = os.path.join(img_folder,image)
			img = cv2.imread(imgr)
			get_and_save_features(img,names)
	print(" ")
	print("CSV file generated successfully.")
	
dataset_path = 'Training_Dataset'
dataset_csv = 'features.csv'

# 68 point face landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# detects face 
detector = dlib.get_frontal_face_detector()


if os.path.exists('features.csv') == True :
	msg = int(input("Dataset csv file already present.Press 1 for using the existing csv file or 2 to create a new csv file for training :"))
	if msg == 2:
		prepare_features_csv()
else:
	prepare_features_csv()


