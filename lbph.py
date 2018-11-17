import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
#from AudioIO import listen, speak
#from settings import veronica_notify
sys.tracebacklimit=None
name=""
names=[]
faces=[]
i=1
c=0
count=0
labels=[]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def test():
    global face_recognizer
    faces,labels=prepare_training_data()
    j=1
    while j<len(faces):
        s=str(faces[j])
        if s[1][0]=="0":
#print (str(j))
            del faces[j]
            j=j-1
        j=j+1
   # except:
      #  print("done")
    print(len(faces))
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    cam=cv2.VideoCapture(0)
    time.sleep(2)
    r,img=cam.read()
    del(cam)
    cv2.imwrite("/home/duke_senior/Desktop/test/t.jpg",img)
    predict()
    

def train():
#    speak("training")
 #   veronica_notify("Training")
	cam=cv2.VideoCapture(0)
	fx=open("counter.txt", "r")
	f=fx.readlines()
	x=int(f[0].strip())
    #time.sleep(10)
	for i in range(20):
		r,img=cam.read()
		cv2.imwrite("/home/duke_senior/Desktop/s2/"+str(x)+".jpg",img)
		x=x+1
		time.sleep(1)
	fx.close()
	fx=open("counter.txt", "w")
	fx.write(str(x))
	del(cam)
    #exit()

def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('/home/duke_senior/Desktop/lbpcascade_frontalface_improved.xml')
    
#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
 
#if no faces are detected then return original img
    if (len(faces) == 0):
        return np.zeros((2,2),dtype='int')
    (x, y, w, h) = faces[0]
 
#return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data():
        """dirs = os.listdir(data_folder_path)
        for dir_name in dirs:
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name"""
        faces = []
        labels = []
        j=0
        fx=open("counter.txt", "r")
        f=fx.readlines()
        x=int(f[0].strip())
        for j in range(x-1):
            image_path = "/home/duke_senior/Desktop/s2/" + str(j)+".jpg"
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            faces.append(face)
            labels.append(int(j/20)+1)
            """cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()"""
        return faces, labels

def predict():
    img=cv2.imread("/home/duke_senior/Desktop/test/t.jpg")
    face, rect = detect_face(img)
    try:
      label,s= face_recognizer.predict(face)
    except Exception as inst:
      print("this is error")
      login()
    fx=open("name.txt", "r")
    cont=fx.readlines()
    print(cont[label-1])
 #   veronica_notify("welcome" + name)
  #  speak("welcome " + name)
    exit()

def login():
	f=input("1. login\n2.train\n0.Exit")
	if(f=="1"):
		test()
	if(f=="2"):
		pswd=input("password:")
		if(pswd=="admin123"):
			fx=open("name.txt", "a")
			fx.write(name+"\n")
			train()
		else:
			exit()
	if(f=="0"):
		exit()


name=input("input your name")
login()
fx=open("name.txt", "a")

