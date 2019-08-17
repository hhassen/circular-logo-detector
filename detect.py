# Object Detector
# Developed by Hassen HARZALLAH : November 2018
#
# Developped on : Python 3.6.5.final.0 (Conda 4.5.11), OpenCV 3.4.1, Numpy 1.14.3
# The programs first extracts the circles (Hough Transform) on each frame,
# then compares each circle with the object using the SIFT detector.
# Execute as follows : detect.py -i positive.avi -o export.csv

import cv2
import numpy as np
import csv
import sys


# Extract circle array from an image
def image_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=70, param2=30,
                               minRadius=1, maxRadius=100)
    return circles


# draw circles from circle array on an image
def draw_circles(image, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)


# draw one circle from propreties array [x,y,radius] on an image
def draw_circle(image, circle):
    if circle is not None:
        circle = np.uint16(np.around(circle))
        i = circle
        center = (i[0], i[1])
        # circle center
        cv2.circle(image, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(image, center, radius, (255, 0, 255), 3)


# draw a bounding box using the circle properties
def draw_box(image, circle):
    if circle is not None:
        circle = np.uint16(np.around(circle))
        i = circle
        xc, yc = (i[0], i[1])
        radius = i[2]
        # draw box
        cv2.rectangle(image, (xc - radius, yc - radius), (xc + radius, yc + radius), (0, 255, 0), 3)


# crop image and avoid overpassing the limits
def image_crop(image, y, x, r):
    y1, y2, x1, x2 = y - r, y + r, x - r, x + r
    if x1 < 0:
        x1 = 0
    if x2 > image.shape[0]:
        x2 = image.shape[0]
    if y1 < 0:
        y1 = 0
    if y2 > image.shape[1]:
        y2 = image.shape[1]
    crop_img = image[x1:x2, y1:y2]
    return crop_img


# return the number of matches between the keypoints of an image and the keypoints entered
def matches_number(sift, img, kp1, des1):
    # kp1 : Keypoints of positive image
    # des1 : descriptors of positive image

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img, None)

    # If there is no keypoints and decsriptors
    if not kp1 or not kp2:
        return None

    if len(kp1) <= 2 or len(kp2) <= 2:
        return None

    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    if len(des1 >= 2) and len(des2) >= 2:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    distance_min = 0.65
    for m, n in matches:
        if m.distance < distance_min * n.distance:
            good.append(m)
    return len(good)


# initialize csv file and erase old content
def csv_initialize(file):
    with open(file, mode='w') as csv_file:
        csv.writer(csv_file, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)


# add a row at the end of the csv file
def csv_addrow(file, circle, frameid):
    circle = np.uint16(np.around(circle))
    i = circle
    xc, yc = (i[0], i[1])
    radius = i[2]
    with open(file, mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frameid, xc - radius, yc - radius, 2 * radius, 2 * radius])


# Interface management
def interface():
    if not len(sys.argv) == 5 or not sys.argv[1] == "-i" or not sys.argv[3] == "-o":
        raise Exception("Interface Error ! Use the following format : detector.py -i positive.avi -o export.csv")
    return str(sys.argv[2]), str(sys.argv[4])


# ***********
# MAIN PROGRAM
# ***********
# ***********
video_file, export_file = interface()
cam = cv2.VideoCapture(video_file)
positive = cv2.imread('positive.png')
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(positive, None)  # calculates the keypoints and the descriptors of the positive image
frameId = 0  # the current frame
csv_initialize(export_file)
# Parameters
threshold_value = 70
NB_matches_min = 7

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
# Loop of the video frames
while True:
    frameId = frameId + 1
    ret, image = cam.read()
    # Thresholding image
    retval, image_r = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    circles = image_circles(image_r)  # Play on : image or image_r : to activate or disable thresholding

    # if we have circles in frame
    if np.count_nonzero(circles) != 0:
        # Loop on the different circles
        for circle in circles[0, :]:
            x, y, r = circle.astype(int)
            crop_img = image_crop(image, x, y, r)
            NB_matches = matches_number(sift, crop_img, kp1, des1)
            print("number of matches :", NB_matches)
            if NB_matches is not None:
                # if we have enough matches draw the box and add the coordinates to the export file
                if NB_matches > NB_matches_min:
                    draw_box(image, circle)
                    csv_addrow(export_file, circle, frameId)

    # write the flipped frame
    out.write(image)
    # draw_circles(image,circles) #to draw all circles given by hough transform
    cv2.imshow('result', image)
    if cv2.waitKey(10) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
