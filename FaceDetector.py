import cv2
import sys
import os
import numpy as np
import json

def facedetection():
    
    idir = sys.argv[1]
    i_files=os.listdir(idir)
    ifiles_path=[os.path.join(idir,efile) for efile in i_files]
    
    fb_list=[]
        
    for efile,efpath in zip(i_files,ifiles_path):
        image = cv2.imread(efpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_cascade_face = cv2.CascadeClassifier('./Pretrainedclassifiers/haarcascade_frontalface_default.xml')
        
        faces_found = haar_cascade_face.detectMultiScale(gray,scaleFactor=1.19,minNeighbors=4)
        
        for (x,y,w,h) in faces_found:
            fb_list.append({"iname": efile, "bbox": [int(x),int(y), int(w), int(h)]})
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
       

    #Storing found face params in json file
    output_json = os.path.join(idir,"results.json")
    #dump list to results.json
    with open(output_json, 'w') as f:
        json.dump(fb_list, f)
        
if __name__ == "__main__":
    facedetection()
