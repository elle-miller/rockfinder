"""
Ikaroa Space Robotics Project

Center of gravity vs center of mass
colour, texture, point cloud ~ bumpy/porous
texture - visual features
different patterns of rock, layers on a piece of wood
contrast ratio/Canny algorithm
Fourier-Mellin transform
if you can a FT of a 2D image, if every thing is same colour/ then everything low frequency
MT - which orientation (horizontal or vertical)

"""
import cv2
import numpy as np
from rawFile import getimage
import time


def main():
    # Continually image and move until at a certain distance away from object
    next_target = 10  # Random start value
    tol = 1

    # Take another image from realsense camera
    # result = getimage()
    # if result is None:
    #     exit(0)

    image_path = "images/"
    image_name = "20201104-184315_col"
    image_type = ".png"
    image_filename = image_path + image_name + image_type
    path = 'cvimages/'
    im = cv2.imread(image_filename)

    # PART ZERO - SEE IF THERE IS A GREEN OBJECT
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Green mask
    mask = cv2.inRange(hsv, (30, 50, 50), (80, 255, 150))
    imask = mask > 0
    green = np.zeros_like(im, np.uint8)
    green[imask] = im[imask]
    green_image = path + image_name + "_greencont.png"
    # cnts, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(im, cnts[0], 0, (0, 255, 0), 2)
    # cv2.imwrite(green_image, green)


    # PART ONE - OBJECT DETECTION
    # Convert BGR image to greyscale
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filtering (5x5 kernel), then perform Otsu thresholding
    blur = cv2.GaussianBlur(im_grey, (5, 5), 0)
    ret, im_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_filename = path + image_name + "_thresh.png"
    cv2.imwrite(processed_filename, im_thresh)

    # Find edges using Canny algorithm on the thresholded image
    im_canny = cv2.Canny(im_thresh, 0, 255, L2gradient=True)

    # Save Canny image
    processed_filename = path + image_name + "_canny.png"
    cv2.imwrite(processed_filename, im_canny)

    # PART TWO - DECISION ALGORITHM
    # Finding Contours
    font = cv2.FONT_HERSHEY_COMPLEX
    img2 = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    cnts, _ = cv2.findContours(im_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    n = len(cnts)
    print("Number of Contours found = " + str(n))

    # Set parameter filters
    min_area = 1000
    max_area = 100000
    min_circle = 0
    max_circle = 1

    # Going through every contours found in the image.
    area = n*[0]
    perimeter = n*[0]
    circularity = n*[0]
    convexity = n*[0]
    inertia = n*[0]
    cX = np.array(n*[0]).astype(int)
    cY = np.array(n*[0]).astype(int)
    i = 0
    j = 0
    for i in range(n-1):
        c = cnts[i]
        M = cv2.moments(c)
        if M["m00"] != 0:

            # Check to see if the center points are close to another pair
            cX[i] = int(M["m10"] / M["m00"])
            cY[i] = int(M["m01"] / M["m00"])

            # Make sure this is not a double up
            if (i > 0) & (abs(cX[i]-cX[i-1]) < 150.):
                pass
            # Filter by restraints
            elif (cv2.contourArea(c) < min_area) | (cv2.contourArea(c) > max_area):
                pass
            else:
                # Compute blob properties
                area[j] = cv2.contourArea(c)
                perimeter[j] = cv2.arcLength(c, True)
                circularity[j] = 4 * np.pi * area[j] / perimeter[j] ** 2
                convexity[j] = cv2.isContourConvex(c)
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                inertia[j] = float(MA) / ma
                cv2.drawContours(img2, [c], 0, (0, 255, 0), 2)
                cv2.circle(img2, (cX[i], cY[i]), 7, (255, 255, 255), -1)
                #cv2.putText(img2, "a: %.0f" % area[j], (cX[i] - 20, cY[i] - 20), font, 1, (255, 255, 255), 2)
                #cv2.putText(img2, "c: %.2f" % circularity[j], (cX[i] - 20, cY[i] - 50), font, 1, (255, 255, 255), 2)
                cv2.putText(img2, "i: %.2f" % inertia[j], (cX[i] - 20, cY[i] - 20), font, 1, (255, 255, 255), 2)
                j = j + 1

    print("True number of contours = " + str(j))

    processed_filename = path + image_name + "_fullcontour.png"
    cv2.imwrite(processed_filename, img2)




    # params = cv2.SimpleBlobDetector_Params()
    #
    # # params.minThreshold = 10
    # # params.maxThreshold = 200
    # params.filterByArea = True
    # params.maxArea = 40000
    # params.minCircularity = 0.1
    # params.minConvexity = 0.1
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(im_canny)
    # next_target = keypoints[-1].pt
    # print("Largest object X = %.2f" % next_target[0])
    # print("Largest object Y = %.2f" % next_target[1])
    #
    # # Draw detected blobs as red circles and show
    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", im_with_keypoints)
    #
    # processed_filename = path + image_name + "_processed.png"
    # cv2.imwrite(processed_filename, im_with_keypoints)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
