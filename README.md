# Face-Detection-and-Emotion-Recognition
This repository conatins the implementation of designed Emotion Recognition model trained using Cohn-Kanade and fer-2013 datasets.The Best Accuracy obtained is ***78.9%***. 

## About the dataset
I have used two datasets - Extended Cohn-Kanade Dataset(CK+) and fer-2013 dataset

# Steps for using the software

Follow the following steps sequentially :-

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
This will perform realtime prediction from the video feed acquired from the webcam.
