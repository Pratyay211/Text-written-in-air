import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
import pickle
import argparse
import keras
import numpy as np
import tensorflow
import argparse
from keras.models import model_from_yaml
import re
import base64
import cv2

def completeWhite(column):
    for i in range(len(column)):
        if column[i]!=255:
            return False
    return True

def isValid(img,start,end):
    for i in range(start,end+1):
        for j in range(len(img[i])):
            if img[i][j]!=255:
                return True
    return False

def predict(bin_dir,img):
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('%s/model.h5' % bin_dir)
    mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))
    img = np.invert(img)
    cv2.imwrite("test.jpg",img)
    img = cv2.resize(img,(28,28))
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = img/255
    output = model.predict(img)
    out = chr(mapping[(int(np.argmax(output, axis=1)[0]))])
    return out
 
def segment(bin_dir):
    img = cv2.imread("ok.jpg",0)
    img = cv2.transpose(img)
    height,width = img.shape[:2]
    startcrop = []
    endcrop = []
    mode = 0
    for i in range(len(img)):
        if(completeWhite(img[i])):
            if mode != 0:
                endcrop.append(i)
            mode = 0
        else:
            if mode == 0:
                startcrop.append(i-5)
            mode = 1
    s = ""
    print(startcrop)
    print(endcrop)
    for i in range(len(startcrop)):
        if(isValid(img,startcrop[i],endcrop[i])):
            newimg = cv2.transpose(img)
            newimg = newimg[0:height,startcrop[i]:endcrop[i]]
            if i == 1:
                cv2.imwrite("x.jpg",newimg)
            temp = str(predict(bin_dir,newimg))
            s = s + temp
            # print(temp)
    # print(s)
    return s

def main():
    return segment('../bin')

if __name__ == '__main__':
    main()
