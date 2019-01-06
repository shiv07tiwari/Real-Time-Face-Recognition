import cv2
import numpy as np

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
count=0
face_data = []
dataset_path = './facedata/'
print("Enter name of the person : ")
file_name = input()
while True:

	ret,frame = cam.read()

	#Generally use Gray Frame to save memory
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if (ret==False):
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	#Select largest face on screen
	for face in faces[-1:]:
		x,y,w,h = face 
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		#Get Region of intrest
		offset = 10
		#Cut out the face with padding, Y_X by default
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] 
		face_section = cv2.resize(face_section,(100,100))
		cv2.imshow("Face",face_section)
		count+=1
		if(count%9==0):
			face_data.append(face_section)
			print(len(face_data))
	cv2.imshow("Image",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed==ord('q')):
		break

#Convert facelist to numpy array
if(len(face_data)==0):
	print("No Data Collected")
	exit()
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save the face data
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Saved !")
cam.release()
cv2.destroyAllWindows()