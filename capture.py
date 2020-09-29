#capture
import cv2
import argparse
import time
import os
import modelupd
import glob
import random
import eel
#import winsound #on your need

frequency=2500
duration=1000

eel.init('WD_INNOVATIVE')
emotions=["angry", "happy", "sad", "neutral"] #default emotion
fishface = cv2.face.FisherFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX


parser=argparse.ArgumentParser(description="Options for emotions based music player (Updating the model)")
parser.add_argument("--update", help="Call for taking new images and retraining the model.", action="store_true")
args=parser.parse_args()    
facedict={}
video_capture=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice=clahe_image[y:y+h, x:x+w]
        faceslice=cv2.resize(faceslice, (350, 350))
        facedict["face%s" %(len(facedict)+1)]=faceslice
    return faceslice

def recface():
    ret, frame=video_capture.read()
    cv2.imwrite('test.jpg', frame)
    cv2.imwrite("images/main%s.jpg" %count, frame)
    gray=cv2.imread('test.jpg',0)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image=clahe.apply(gray)
    return clahe_image

def recface2():
    clahe_image=recface()
    face=facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face)>=1:
        faceslice=crop(clahe_image, face)
        #return faceslice
    else:
        print("No/Multiple faces detected!!, passing over the frame")

def recface3(emotion):
    print("\n\nLook "+emotion+" untill the timer expires and keep the same emotion for some time.")
    print('\a')
    
    
    for i in range(0, 5):
        print(5-i)
        time.sleep(1)
    
    while len(facedict.keys())<16:
        recface2()

    for i in facedict.keys():
        path, dirs, files = next(os.walk("dataset/%s" %emotion))
        file_count = len(files)+1
        cv2.imwrite("dataset/%s/%s.jpg" %(emotion, (file_count)), facedict[i])
    facedict.clear()

def modelupd(emotions):
    print("Update mode for model is ready")
    dircheck(emotions)
    
    for i in range(0, len(emotions)):
        recface3(emotions[i])
    print("Collected the images, looking nice! Now updating the model...")
    modelupd.update(emotions)
    print("Model train successful!!")

def dircheck(emotions):
    for emotion in emotions:
        if os.path.exists("dataset/%s" %emotion):
            pass
        else:
            os.makedirs("dataset/%s" %emotion)

def emocheck():
    prediction=[]
    confidence=[]

    for i in facedict.keys():
        pred, conf=fishface.predict(facedict[i])
        cv2.imwrite("images/%s.jpg" %i, facedict[i])
        prediction.append(pred)
        confidence.append(conf)
    output=emotions[max(set(prediction), key=prediction.count)]    
    print("You seem to be %s" %output) 
    facedict.clear()
    return output;
count=0
#Careful here
@eel.expose
def getEmotion():
   
    count=0
    while True:
        count=count+1
        recface2()
        if args.update:
            modelupd(emotions)
            break
        elif count==10:
            fishface.read("model.xml")
            return emocheck()
            break

#eel.start('main.html', options=web_app_options)
#options={'host':'file', 'port': '//'}
eel.start('main.html')


