# importing necessary packages
import os
import shutil
from tqdm import tqdm
import cv2
import dlib

# generating bounding boxes coordinates
def compute_bounding_boxes_dimensions(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

# delete Training dataset if it already exists
def delete_prepared_dataset():

	# checks if the training dataset already exists
	if os.path.exists('Training_Dataset') == True :

		# deletes folder and all its contents
		shutil.rmtree('Training_Dataset')

# remove bad images and get the relevant images for training
def get_relevant_images(image_list,number_of_images):
	
	# exculding the first few images of each subjects as they are irrelevant for the training of the model
	bad_image_count = int((percent_irrelevant_data/100) * number_of_images)
	image_list.sort(key = lambda x: (x.split('_')[2]).split('.')[0])
	image_list = image_list[bad_image_count:]
	relevant_image_list.extend(image_list)

# create the Training dataset directory	
def create_dataset_structure(string_1,string_2,data,image):
	image_name = image.split('/')[4]
	if string_1 == string_2 :
		if os.path.exists('Training_Dataset') == False :
			os.mkdir('Training_Dataset')
		if os.path.exists('Training_Dataset/'+ data) == False :
			os.chdir('Training_Dataset')
			os.mkdir(data)
			os.chdir('..')
		if image_name in relevant_image_list :
			shutil.copy(image,'Training_Dataset/' + data)	
# data Preparation			
def Prepare_Cohn_Kanade_dataset(Dataset_path):
	subfolder_image_list = []
	for folder in os.listdir(Dataset_path):
		if folder == 'cohn-kanade-images' :
			images_folder = os.path.join(Dataset_path,folder)
		elif folder == 'Emotion':
			labels_folder = os.path.join(Dataset_path,folder)

	for labels_subfolders in tqdm(os.listdir(labels_folder)):
		labels_subfolder_path = os.path.join(labels_folder,labels_subfolders)

		for label_files in os.listdir(labels_subfolder_path):
			label_files_path = os.path.join(labels_subfolder_path,label_files)

			# used for matching strings with the labels csv of cohn-Kanade dataset
			string_2 = labels_subfolders + '/' + label_files + '/'

			for label_text in os.listdir(label_files_path):
				label_text_file = os.path.join(label_files_path,label_text)

				# used for naming the images in their respective label folders
				with open (label_text_file, "r") as myfile:
  			  		data=str(int(float(myfile.read().strip())))

				for images_subfolders in os.listdir(images_folder):
					image_subfolder_path = os.path.join(images_folder,images_subfolders)

					for image_files in os.listdir(image_subfolder_path):
						image_files_path = os.path.join(image_subfolder_path,image_files)
					
						# used for matching strings with the labels csv of cohn-Kanade dataset
						string_1 = images_subfolders + '/' + image_files + '/'

						subfolder_image_list.clear()

						for imgs in os.listdir(image_files_path):
							image = os.path.join(image_files_path,imgs)
							create_dataset_structure(string_1,string_2,data,image)
							subfolder_image_list.append(imgs)

						get_relevant_images(subfolder_image_list,len(os.listdir(image_files_path)))

# deleting the labels that are not required in ths model
def delete_unrequired_labels():
	if os.path.exists('Training_Dataset/0'):
		shutil.rmtree('Training_Dataset/2')
	if os.path.exists('Training_Dataset/2'):
		shutil.rmtree('Training_Dataset/2')
	if os.path.exists('Training_Dataset/3'):
		shutil.rmtree('Training_Dataset/3')
	if os.path.exists('Training_Dataset/4'):
		shutil.rmtree('Training_Dataset/4')

# preparing fer 2013 dataset and combing the images with the Cohn-Kanade dataset
def Prepare_fer2013_dataset(Dataset_path):
	dataset_save_location = 'Training_Dataset'
	
	# only the labels - Angry,Happy,Sad and Suprise are used. Rest of the labels are discarded
	required_labels = ['0','3','4','5']
	for files in tqdm(os.listdir(Dataset_path)):
		if files in required_labels:
			label_path = os.path.join(Dataset_path,files)
			
			# matching the labels of Cohn Kanade with fer-2013 and saving the fer 2013 images on the matched label folder of Cohn-Kanade dataset 
			for images in os.listdir(label_path):
				image = os.path.join(label_path,images)
				if files == '0':
					shutil.copy(image,'Training_Dataset/1')
				if files == '3':
					shutil.copy(image,'Training_Dataset/5')
				if files == '4':
					shutil.copy(image,'Training_Dataset/6')
				if files == '5':
					shutil.copy(image,'Training_Dataset/7')

# Renaming labels to suitable names
def rename_labels():
	shutil.move('Training_Dataset/5','Training_Dataset/2')
	shutil.move('Training_Dataset/6','Training_Dataset/3')
	shutil.move('Training_Dataset/7','Training_Dataset/4')

# cropping faces and resizing them 
def resize_all_images():
	for labels in tqdm(os.listdir('Training_Dataset')):
		label_path = os.path.join('Training_Dataset',labels)
		for images in os.listdir(label_path):
			image = os.path.join(label_path,images)
			img = cv2.imread(image)
			img_resized = cv2.resize(img,image_size)
			gray_img = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
			rects = detector(gray_img)
			for rect in rects:
				(x,y,w,h) = compute_bounding_boxes_dimensions(rect)
				if x > 0 and y > 0 and w > 0 and h > 0:
					# cropping and resizing only the faces
					img_resized = cv2.resize(img_resized[y:y+w, x:x+h],image_size)
			# saving cropped and resized images
			cv2.imwrite(image,img_resized)
														
print(" MODIFYING and SIMPLIFYING DATASET....")
print(" ")

# Enter the folder path containing the images dataset and emotion classes
Dataset_path_1 = 'Cohn-Kanade_Dataset'
Dataset_path_2 = 'fer2013_extracted_Dataset'

detector = dlib.get_frontal_face_detector()

# image resize dimensions
image_size = (200,200)

# list to contain all the relevant images names
relevant_image_list = []

# percentage of images to be discarded from each subject's folder in a sequential order
percent_irrelevant_data = 40

# deletes the existing prepared dataset if it exists
delete_prepared_dataset()

# Calling all the necessary function in the correct order
Prepare_Cohn_Kanade_dataset(Dataset_path_1)
delete_unrequired_labels()
Prepare_fer2013_dataset(Dataset_path_2)
rename_labels()
resize_all_images()

print(" ")
print(" MODIFICATION COMPLETE.")
print(" Training DATASET SAVED.")
					
					
							



