import cv2
import numpy as np
import copy
import math
import main as detect
import predict as predict
#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
threshold = 20  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
traverse_point = []
# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
bgModel = None
def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    global traverse_point
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            top = []
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                top.append(start)
            point = min(top,key=lambda item:item[1])
            point = (point[0]+128,point[1])
            if startDetection:
                if(cnt==0):
                    traverse_point.append(point)
                else:
                    traverse_point.append((-1,-1))
            return True, cnt
    return False, 0


startDetection = False
# Camera
def main():
    global startDetection,traverse_point,triggerSwitch,isBgCaptured,learningRate,bgSubThreshold,threshold,blurValue,bgModel
    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
    finale = cv2.imread('template.jpg',0)
    s = None
    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (128, 0),
                    (640,256), (255, 0, 0), 2)
        # print(int(cap_region_x_begin * frame.shape[1])
        # print(frame.shape[1])
        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            img = img[0:256,
                        128:640]  # clip the ROI
            cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('ori', thresh)


            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                isFinishCal,cnt = calculateFingers(res,drawing)
                if triggerSwitch is True:
                    if isFinishCal is True and cnt <= 2:
                        print (cnt)
                        #app('System Events').keystroke(' ')  # simulate pressing blank space
                
            for i in range(0,len(traverse_point)):
                    cv2.circle(frame, traverse_point[i], 2, [0, 0, 255], -1)
                    # cv2.circle(finale,(int((traverse_point[i][0]-128)),int(traverse_point[i][1]/2)), 2, [0, 0, 0], -1)
                    # print(traverse_point[i])
            for i in range(1,len(traverse_point)):
                    if(traverse_point[i][0]!=-1 and traverse_point[i-1][0]!=-1):
                        cv2.line(frame,traverse_point[i],traverse_point[i-1],[0,255,0],15,lineType = cv2.LINE_AA)
                        cv2.line(finale,(int((traverse_point[i][0]-128)),int(traverse_point[i][1]/2)),(int((traverse_point[i-1][0]-128)),int(traverse_point[i-1][1]/2)),[0,0,0],15,lineType = cv2.LINE_AA)
                    else:
                        i = i+1
            cv2.imshow('output', drawing)
        if s != None:
            cv2.putText(frame,s, (0, 480), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), lineType=cv2.LINE_AA)
        cv2.imshow('original', frame)
        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            cv2.imwrite("ok.jpg",finale)
            finale = cv2.imread("template.jpg",0)
            if s == None:
                s = predict.main()
            else:
                s = s + " " + predict.main()
            # cv2.putText(frame,s, (0,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            traverse_point = []
            startDetection = False
            print ('!!!Reset BackGround!!!')
        elif k == ord('n'):
            triggerSwitch = True
            print ('!!!Trigger On!!!')
        elif k == ord('d'):
            startDetection = True
            print ('!!!!Detection On!!!!')

if __name__ == '__main__':
    	main()