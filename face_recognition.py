import numpy as np
import cv2
import os

def dist(x1,x2):
	return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,querypoint,k=3):
	distances = []
	m = X.shape[0]

	for i in range(m):
		d = dist(querypoint,X[i])
		distances.append((d,Y[i]))

	distances = sorted(distances) 
	distances = distances[:k]
	distances = np.array(distances)
	return distances



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
print("Done")
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
count=0
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
		#print(face_section)
		face_section = cv2.resize(face_section,(100,100))
		
		count+=1
		if(count%9==0 or count==1):
			sample_face_data.append(face_section)
	
	cv2.imshow("Image",frame)
	key_pressed = cv2.waitKey(1)  & 0xFF
	if(key_pressed==ord('q')):
		break
	if(len(sample_face_data)==0):
		continue
	sample_face = np.asarray(sample_face_data)
	sample_face = sample_face.reshape((sample_face.shape[0],-1))

	for i in range(sample_face.shape[0]):
		vals = knn(X,Y,sample_face[i])
		print(vals)
		new_vals = np.unique(vals[:,1],return_counts=True)
		print(new_vals)
		max_freq_index = new_vals[1].argmax()
		print(max_freq_index)
		prediction = new_vals[0][max_freq_index]
		print(names[prediction])
# new_vals = np.unique(vals[:,1],return_counts=True)

# max_freq_index = new_vals[1].argmax()

# prediction = new_vals[0][max_freq_index]