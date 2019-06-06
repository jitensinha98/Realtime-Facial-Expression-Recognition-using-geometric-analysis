# Face-Detection-and-Emotion-Recognition
This repository conatins the implementation of designed Emotion Recognition model trained using ***Cohn-Kanade*** and ***fer-2013*** datasets.The Best Accuracy obtained is ***78.9%***. 

## About the datasets
I have used two datasets - Extended Cohn-Kanade Dataset(CK+) and fer-2013 dataset .  The CK+ dataset contains only 123 subjects whereas the fer-2013 dataset contains more than thousand subjects. As a result The test accuracy of CK+ is ***92.3%*** whereas the test result of of fer-2013 is below ***70%*** . So i have performed a combination of both the dataset to get a more balanced dataset . I also had to discard 40% of images from all the subjects in Cohn-Kanade's dataset because they were meaningless and irrelevant. Details of acquiring CK+ dataset is [here](http://www.pitt.edu/~emotion/ck-spread.htm) and fer-2013 is [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) . 

## Modules Used
- Keras
- Tensorflow-gpu == 1.5
- tqdm
- dlib
- opencv
- os
- shutil
- pandas
- numpy
- sklearn
- imutils

# Steps for using the software

Follow the following steps sequentially :-

- Download ***shape_predictor_68_face_landmarks.dat*** from [here](https://github.com/davisking/dlib-models) and put it in the Repository directory.

- Download the datasets from the above links and put their contents in the  folders - Cohn-Kanade_Dataset and fer2013 respectively.

- Run ***Extract_fer2013_images.py*** 
```
python3 Extract_fer2013_images.py
```
It will extract the images from csv file in fer-2013 dataset and create a seperate folder named ***fer2013_extracted_Dataset*** containing all the images in the respective label named subfolders.

- Run ***Data_preparation.py***
```
python3 Data_preparation.py
```
It will combine the two datasets along with cropping and resizing each of the image in both the datasets so that only face portion pixels are visible.It will create a folder containing modified images named ***Training_Dataset*** . We also eliminate the first 40 % of the images of each subjects in the Cohn-Kanade dataset because they are irrelevant.

- Run ***Generate_features_csv.py***
```
python3 Generate_features_csv.py
```
This will apply the 68 point landmark system on the ***Training_Dataset*** images and find euclidean distances between each points and store them as feature on a csv file names ***features.csv***.Total features would be 68x68 = 4642 features + 1 target label.

- Run ***Training_Classifier.py***
```
python3 Training_Classifier.py
```
An ANN is designed using Keras for predicting the target variables using the features.The model will be saved on path ***Saved_Model/Classifier_model.sav***

- Run ***face_detection_and_emotion_recognition.py***
```
python3 face_detection_and_emotion_recognition.py
```
This will perform realtime prediction from the video feed acquired from the webcam.

## About the Software
I have dlib face detector and dlib 68 point shape predictor for Emotion Recognition . The predictor provides 68 coordinates in the face to predict its shape . I have used the euclidean distances among these points as features to predict target variable . While computing realtime predictions i have cropped and resize the face portion of every face for accurate predictions over long distances from the video source . I have also designed a progress bar which displays the probability of each of the four emotions - ***Happy,Sad,Angry,Suprise***. Adding any more emotions causes a drop in the accuracy so i have used only these four emotions for training.

## Sample Face Landmark on Dataset Images

![picture](https://github.com/jitensinha98/Face-Detection-and-Emotion-Recognition/blob/master/Images/1.jpg)

![picture](https://github.com/jitensinha98/Face-Detection-and-Emotion-Recognition/blob/master/Images/2.jpg)

![picture](https://github.com/jitensinha98/Face-Detection-and-Emotion-Recognition/blob/master/Images/3.jpg)


