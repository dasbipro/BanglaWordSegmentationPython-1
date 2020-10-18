from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
# from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *
import nms

def main():
    print('Hello AI')
    image = cv2.imread('Data/image.jpg')
    #cv2.imshow("image",image)
    #cv2.waitKey()
    
    # define parameters of HOG feature extraction
    size = (64,128)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = .3
    data = []
    labels = []

    pos_im_path = r"./Data/SmallData/positive" 
    neg_im_path = r"./Data/SmallData/negative"

    pos_im_listing = os.listdir(pos_im_path) 
    neg_im_listing = os.listdir(neg_im_path)
    num_pos_samples = len(pos_im_listing) 
    num_neg_samples = len(neg_im_listing)
    print(num_pos_samples,num_neg_samples)
    
    le = LabelEncoder()

    for file in pos_im_listing:
        img = cv2.imread(pos_im_path + '/' + file) # open the file
        img = cv2.resize(img, size)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # calculate HOG for positive features
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
        data.append(fd)
        # cv2.imshow('abcd',fd)
        # cv2.waitKey()
        labels.append(1)
    
    for file in neg_im_listing:
        img= cv2.imread(neg_im_path + '//' + file)
        img = cv2.resize(img, size)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Now we calculate the HOG for negative features
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        data.append(fd)
        labels.append(0)
    
    # encode the labels, converting them from strings to integers
    # labels = le.fit_transform(labels)
    
    print(" Constructing training/testing split...")
    trainData, testData, trainLabels, testLabels = train_test_split(
        data, labels, test_size=0.20, random_state=42)
    
    print(trainData[0])
    print("Label = ", trainLabels[0])

    trainData = np.array(trainData)
    testData = np.array(testData)
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)

    print(" Training Linear SVM classifier..")
    model = LinearSVC()
    model.fit(trainData, trainLabels)
    pred = model.predict(testData)

    print(classification_report(testLabels, pred, labels=model.classes_, target_names=None, sample_weight=None, digits=3))
    print(accuracy_score(testLabels,pred))
    segmentation(model,image)
def seg(model,img):
    im=img
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    ''' the above code uses a pretrained SVM via HOG descriptors provided by the open cv database.
    This database is limited to the training it has performed hence cannot be used in any other angle other than perp. to the centroid
    Thus if you want to implement the HOG + SVM method, you'll have to train your own SVM with your own data'''
    cap= cv2.VideoCapture(0)
    # the above code uses the OpenCV library to capture video frames from the camera: select 0 for the primary pc webcam & 1 for an external camera

    while True:
        #running an infinite loop so that the process is run real time.
        ret, img = cap.read() # reading the frames produced from the webcam in 'img' an then returning them using the 'ret' function.
        found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) # describing the parameters of HOG and returning them as a Human found function in 'found'
        found_filtered = [] #filtering the found human... to further improve visualisation (uses Gaussian filter for eradication of errors produced by luminescence.
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
                else:
                    found_filtered.append(r)
            draw_detections(img, found) # using the predefined bounding box to encapsulate the human detected within the bounding box.
            draw_detections(img, found_filtered, 3) # further filtering the box to improve visualisation.
            print('%d (%d) found' % (len(found_filtered), len(found))) # this will produce the output of the number of humans found in the actual command box)
        cv2.imshow('img', img) # finally showing the resulting image captured from the webcam.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # defining a key to quit and stop all processes. The key is 'q'
    cap.release()
    cv2.destroyAllWindows() 
def segmentation(model,img):
    im=cv2.resize(img,(900,900))
    im = img
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = .3
    min_wdw_sz = (100, 40)
    step_size = (10, 10)

    clf = model
    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=1.25):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            #fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd = hog(img, orientations, pixels_per_cell, cells_per_block, feature_vector=True) 
            print(im.size)
            cv2.imshow('img',img)
            cv2.waitKey()
            if(im.size!=(300,300)):
                continue
            pred = clf.predict(fd.reshape(1,-1))
            if pred == 1:
                print ("Detection:: Location -> ({}, {})".format(x, y))
                print ("Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd)))
                detections.append((x, y, clf.decision_function(fd),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
            # If visualize is set to true, display the working
            # of the sliding window
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _, _  in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                        im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(30)
        # Move the the next scale
        scale+=1

    # Display the results before performing NMS
    clone = im.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
    cv2.imshow("Raw Detections before NMS", im)
    cv2.waitKey()

    # Perform Non Maxima Suppression
    #detections = nms(detections, threshold)

    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey()
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh
if(__name__=='__main__'):
    main()