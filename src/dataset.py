import cv2
import numpy as np
import copy
import math

def main(name):
    filepath = '../data/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.tif'

    img = cv2.imread(filepath,0)
    for i in range(img.shape[0]):
        if(i%100==0):
            cv2.putText(img,str(i),(i,0),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)
        elif(i%20==0):
            cv2.circle(img, i, 2, [0, 0, 255], -1)
    cv2.imshow("preview",img)
    num = int(input("enter number of cuts"))
    xprev = 0
    for i in range(num-1):
        xcur = int(input("Enter current x"))
        frame = cv2.imread(filepath,0)
        frame = frame[0:frame.shape[1],xprev:xcur]
        xprev = xcur
        mewpath = '../newdata/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
        cv2.imwrite(newpath,frame)

if __name__ == '__main__':
	main("")