# importing necessary modules
import pickle
import dlib
from imutils import face_utils
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import pandas as pd
import os

# Training standard scalar model using the features.csv features
def train_Standarad_Scalar(x):
	scaler = StandardScaler()
	scaler.fit(x)
	return scaler

# get label names
def get_progressbar_and_label(pred , progress_bar):

	# finds the index of the array having highest probablilty
	label = pred.index(max(pred)) + 1

	# generating class
	if label == 1 :
		class_ = 'Angry'
	elif label == 2:
		class_ = 'Happy'
	elif label == 3:
		class_ = 'Sad'
	elif label == 4:
		class_ = 'Suprise'

	if progress_bar == True :
		
		# clears terminal screen after each iteration
		os.system('clear')
		
		# used to signify progression
		progress_bar_1 = ['|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|']
		progress_bar_2 = ['|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|']
		progress_bar_3 = ['|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|']
		progress_bar_4 = ['|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|','|']

		# calculating probability percent
		label_1_percent = pred[0] * 100
		label_2_percent = pred[1] * 100
		label_3_percent = pred[2] * 100
		label_4_percent = pred[3] * 100

		# converting percentage into number of (|) 
		progress_1 = int(pred[0] * len(progress_bar_1))
		progress_2 = int(pred[1] * len(progress_bar_2))
		progress_3 = int(pred[2] * len(progress_bar_3))
		progress_4 = int(pred[3] * len(progress_bar_4))

		# used for incrementing or decrementing progress bar
		count_1 = len(progress_bar_1) - 1
		count_2 = count_1
		count_3 = count_1
		count_4 = count_1
	
		# used to generate 'angry' progress bar
		while progress_bar_1.count('|') > progress_1 :
			progress_bar_1[count_1] = ' '
			count_1 = count_1 - 1
		while progress_bar_1.count('|') < progress_1 :
			progress_bar_1[count_1] = ' '
			count_1 = count_1 + 1	
		print("Angry     : [",''.join(progress_bar_1),"] ",label_1_percent,"%");

		# used to process 'Happy' progress bar
		while progress_bar_2.count('|') > progress_2 :
			progress_bar_2[count_2] = ' '
			count_2 = count_2 - 1
		while progress_bar_2.count('|') < progress_2 :
			progress_bar_2[count_2] = ' '
			count_2 = count_2 + 1	
		print("Happy     : [",''.join(progress_bar_2),"] ",label_2_percent,"%");
	
		# used to process 'Sad' progress bar
		while progress_bar_3.count('|') > progress_3 :
			progress_bar_3[count_3] = ' '
			count_3 = count_3 - 1
		while progress_bar_3.count('|') < progress_3 :
			progress_bar_3[count_1] = ' '
			count_3 = count_3 + 1	
		print("Sad       : [",''.join(progress_bar_3),"] ",label_3_percent,"%");

		# used to process 'Suprise' progress bar	
		while progress_bar_4.count('|') > progress_4 :
			progress_bar_4[count_4] = ' '
			count_4 = count_4 - 1
		while progress_bar_4.count('|') < progress_4 :
			progress_bar_4[count_4] = ' '
			count_4 = count_4 + 1	
		print("Suprise   : [",''.join(progress_bar_4),"] ",label_4_percent,"%");
		

	return class_

# get bounding boxes coordinates for plotting
def compute_bounding_boxes_dimensions(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

# Processing video feed 
def Process_Realtime():

	print(" ")
	key = int(input("Press 1 to display emotion status bars v for displaying emotion probablilty else press 2 [Displaying status bars will result in slow prediction] : "))
	if key == 1 :
		progress_bar = True
	else:
		progress_bar = False

	vid = cv2.VideoCapture(0)

	while True : 
	
		# reading frames
		ret,frame = vid.read()

		# converting each frame to grayscale
		gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		rects = detector_1(gray_frame)
		
		# iterating through each face rectangle dimensions
		for rect in rects:
			(x,y,w,h) = compute_bounding_boxes_dimensions(rect)
			
			# dlib detector returns negative co-ordinates for partial faces which are imposiible to resize
			if x > 0 and y > 0 and w > 0 and h > 0 :

				# cropping frame to display only the detected face
				gray_frame_cropped = cv2.resize(gray_frame[y:y+w, x:x+h],frame_size)

				rect_ = detector_2(gray_frame_cropped)

				coordinates_list.clear()
				euclidean_distance_list.clear()
		
				# second detector should return only one face
				if len(rect_) == 1:
					shape = predictor(gray_frame_cropped, rect_[0])
					shape = face_utils.shape_to_np(shape)

					for elements in shape.tolist():
						coordinates_list.append(elements)

					# fig euclidean distances between each landmark points detected
					for i in range(0,len(coordinates_list)):
						for item in coordinates_list:
							euclidean_distance = distance.euclidean(coordinates_list[i], item)
							euclidean_distance_list.append(euclidean_distance)	
	
					euclidean_distance_features = scaler.transform([euclidean_distance_list])
		
					# predicting class
					pred = loaded_model.predict(euclidean_distance_features)

					label = get_progressbar_and_label(pred[0].tolist(),progress_bar)

					# plotiing bounding boxes and labels				
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.putText(frame,label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		
		# displays modified video feed
		cv2.imshow("Output", frame)

		# Stoping video feed
		if cv2.waitKey(1) & 0xFF == ord('a') :
			break;

coordinates_list = []
euclidean_distance_list = []

loaded_model = pickle.load(open('Saved_Model/Classifier_model.sav', 'rb'))

# Using two detectors-one for obtaining a cropped image of face and another for detecting face in the cropped image for 68 pint prediction
detector_1 = dlib.get_frontal_face_detector()
detector_2 = dlib.get_frontal_face_detector()

# used for predicting 68-point face landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frame_size = (200,200)

# using features from the dataset to train standard scalar 
dataset = pd.read_csv('features.csv')
x = dataset[dataset.columns[0:(dataset.shape[1]-1)]].values

scaler = train_Standarad_Scalar(x)
Process_Realtime()



