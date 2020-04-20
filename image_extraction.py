# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:42:40 2020

@author: Gopalji
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/NN/datasets/traffic-signs/Test/images.png',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
im = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [im,img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(7):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

