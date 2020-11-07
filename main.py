"""
Ikaroa Space Robotics Project

"""
import cv2
import numpy as np
from imageprocessing import getImage, getMostCompellingObject
import time


def main():

    # Do you have a realsense camera with you?
    have_cam = False
    if have_cam:
        # Take an image from realsense camera
        image_filename = getImage()
    else:
        image_filename = "images/20201107-155051_col.png"
        # image_filename = "images/20201104-184315_col.png"

    if image_filename is None:
        print("No image received!")
        exit(0)

    # Read image and return pixel coordinates of most compelling objects
    im = cv2.imread(image_filename)
    pixel_coords = getMostCompellingObject(im)
    print("Pixel coords of target: " + str(pixel_coords[0]) + ", " + str(pixel_coords[1]))

    # Transform to an actual position

    # Move to position

    # Pick it up

    # Move to box

    # Place in box


if __name__ == '__main__':
    main()
