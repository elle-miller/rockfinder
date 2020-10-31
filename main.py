"""
Ikaroa Space Robotics Project

"""
import cv2
import numpy as np
from rawFile import getimage


def main():
    # Continually image and move until at a certain distance away from object
    next_target = 10  # Random start value
    tol = 1
    while next_target < tol:
        # Take another image from realsense camera
        getimage()
        filename = "martiansurface.png"
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # Filter by area
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(im)
        next_target = keypoints[0].pt
        print("Largest object X = %.2f" % next_target[0])
        print("Largest object Y = %.2f" % next_target[1])

        # Draw detected blobs as red circles and show
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
