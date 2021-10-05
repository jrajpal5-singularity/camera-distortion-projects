# https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
# https://www.tutorialspoint.com/opencv/opencv_canny_edge_detection.htm
# https://learnopencv.com/geometry-of-image-formation/
# import the necessary packages

from imutils import paths
import numpy as np
import imutils
from mediapipe import python
import cv2


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 5, 125)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.imshow("grayscale",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("grayscale", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # compute the bounding box of the of the paper region and return it
    minsize = cv2.minAreaRect(c)
    print("min area rect is " + str(minsize))
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


# initialize the known distance from the camera to the object, which
# in this case is 21.5 inches

KNOWN_DISTANCE = 21.5
# initialize the known object width, which in this case, the piece of
# paper is 8.3 inches wide

KNOWN_WIDTH = 2.8
# load the furst image that contains an object that is KNOWN TO BE 21.5 inches
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("imagesecond.jpg")
image = cv2. resize(image, (960, 540))
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print("focal length is " + str(focalLength))

# loop over the images
for imagePath in sorted(paths.list_images("images")):
    # load the image, find the marker in the image, then compute the distance to the marker from the camera
    image = cv2.imread(imagePath)
    marker = find_marker(image)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    # draw a bounding box around the image and display it
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fft" % (inches / 12),
                (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)