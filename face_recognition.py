import numpy as np
import cv2
import os

def dist(x1,x2):
	return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,querypoint,k=1):
	distances = []
	m = X.shape[0]

	for i in range(m):
		d = dist(querypoint,X[i])
		distances.append((d,Y[i]))

	distances = sorted(distances) 
	distances = distances[:k]
	distances = np.array(distances)
	new_distances = np.unique(distances[:,1],return_counts=True)
	max_freq_index = new_distances[1].argmax()
	prediction = new_distances[0][max_freq_index]

	return prediction



dataset_path = './facedata/'
face_data = []
face_label = []

id = 0 #ID of exisiting data
names = {} #Dictionary for id- label mapping

#Data Preperation
for x in os.listdir(dataset_path):
	if(x.endswith('.npy')):
		x_name = x.replace(".npy","")
		names[id] = x_name
		data_item = np.load(dataset_path+x)
		face_data.append(data_item)

		id_label = id*np.ones((data_item.shape[0],))
		id += 1
		face_label.append(id_label)

X = np.concatenate(face_data,axis=0)
Y = np.concatenate(face_label, axis=0)

#Read and Predict
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
sample_face_data = []
while True:

	ret,frame = cam.read()
	if(ret==False):
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		
		offset = 10
		
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] 
		
		face_section = cv2.resize(face_section,(100,100))
		
		prediction = knn(X,Y,face_section.flatten())
		cv2.putText(frame,names[prediction],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	
	cv2.imshow("Image",frame)
	key_pressed = cv2.waitKey(1)  & 0xFF
	if(key_pressed==ord('q')):
		break

