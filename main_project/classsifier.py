from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog

import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#import pickle5 as pickle
import pickle
filename = 'BALANAGAR_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
classifier=loaded_model


def Classification(path):
    global y_test
    img = path
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    temp = []
    temp.append(segmented_data.ravel())
    temp = np.array(temp)
    predict = classifier.predict(temp)[0]
    img = path
    img = cv2.resize(img, (500, 500))
    name = fruits_names[predict]
    return name,predict
    #if name is not None:
    #    cv2.putText(img, 'Fruit classify as ' + fruits_names[predict] +" " + name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    # 0.6, (0, 255, 255), 2)
    # cv2.imshow("Classification Result : " + 'Fruit classified as ' + fruits_names[predict] + " " + name, img)




fruits_names = ['BALANAGAR 0%', 'BALANAGAR 25%', 'BALANAGAR 50%', 'BALANAGAR 75%', 'BALANAGAR 100%']



# vid = cv2.VideoCapture(0)
  
# while(True):

#     ret, frame = vid.read()
#     name,predict = Classification(frame)
#     if name:
#         cv2.putText(frame, fruits_names[predict] +" " + name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
#                     2.5, (0, 255, 255), 2)


#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()



frame = cv2.imread('25.jpg')
name,predict = Classification(frame)
#if name:
 #   cv2.putText(frame, 'Fruit classify as ' + fruits_names[predict] +" " , (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
  #                  2.5, (0, 255, 255), 2)
if name=="BALANAGAR 0%":
    
    a="TSS=8-10"
    b="HARVEST AFTER 15-20 DAYS"
elif name=="BALANAGAR 25%":
    a="TSS=10-13"
    b="HARVEST AFTER 8-10 DAYS"
elif name=="BALANAGAR 50%":
    a="TSS=13-15"
    b="HARVEST FRUIT(LONG TERM STORAGE)"
elif name=="BALANAGAR 75%":
    a="TSS=14-17"
    b="HARVEST FRUIT(SHORT TERM STORAGE)"
elif name=="BALANAGAR 100%":
    a="TSS=16-19"
    b="HARVEST FRUIT(FOR LOCAL MARKET CONSUMPTION)"
    
cv2.putText(frame, 'Fruit classify as ' + fruits_names[predict] , (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
cv2.putText(frame,  a , (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
cv2.putText(frame, b, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

cv2.imshow('frame', frame)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()