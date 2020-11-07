import cv2
import numpy as np
import time
import pyrealsense2 as rs

"""
Elle's notes:
Center of gravity vs center of mass
colour, texture, point cloud ~ bumpy/porous
texture - visual features
different patterns of rock, layers on a piece of wood
contrast ratio/Canny algorithm
Fourier-Mellin transform
if you can a FT of a 2D image, if every thing is same colour/ then everything low frequency
MT - which orientation (horizontal or vertical)
"""

image_path = "images/"
image_name = "20201104-184315_col"  # First rock image
image_name = "20201107-155051_col"  # Green leaf image
image_type = ".png"
image_filename = image_path + image_name + image_type
path = 'cvimages/'
font = cv2.FONT_HERSHEY_COMPLEX


def getMostCompellingObject(im):
    """
    getMostCompellingObject

    Finds the most interesting object from an image, using the following priority
    1. Green objects
    2. Textured objects
    :param im: Colour image from RealSense camera
    :return: Pixel coordinates of most compelling object
    """
    # PART ZERO - SEE IF THERE IS A GREEN OBJECT
    pixel_coord = isThereLife(im)
    if pixel_coord is not None:
        return pixel_coord

    # PART ONE - IF NO PLANT, FIND NEXT MOST INTERESTING THING
    img_col = im.copy()  #cv2.imread(image_filename, cv2.IMREAD_COLOR)
    im_canny = getCannyFromBGR(im)

    # Find contours
    cnts, _ = cv2.findContours(im_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    n = len(cnts)
    print("Number of Contours found = " + str(n))

    # Set contour filters
    min_area = 1000
    max_area = 100000
    min_circle = 0
    max_circle = 1

    # Going through every contours found in the image.
    area = n * [0]
    perimeter = n * [0]
    circularity = n * [0]
    convexity = n * [0]
    inertia = n * [0]
    cx = np.array(n * [0]).astype(int)
    cy = np.array(n * [0]).astype(int)
    i = 0
    j = 0
    for i in range(n - 1):
        c = cnts[i]
        M = cv2.moments(c)
        if M["m00"] != 0:

            # Check to see if the center points are close to another pair
            cx[i] = int(M["m10"] / M["m00"])
            cy[i] = int(M["m01"] / M["m00"])

            # Make sure this is not a double up
            if (i > 0) & (abs(cx[i] - cx[i - 1]) < 150.):
                pass
            # Filter by restraints
            elif (cv2.contourArea(c) < min_area) | (cv2.contourArea(c) > max_area):
                pass
            else:
                # Compute contour properties if pass all tests
                area[j] = cv2.contourArea(c)
                perimeter[j] = cv2.arcLength(c, True)
                circularity[j] = 4 * np.pi * area[j] / perimeter[j] ** 2
                convexity[j] = cv2.isContourConvex(c)
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                inertia[j] = float(MA) / ma
                cv2.drawContours(img_col, [c], 0, (0, 255, 0), 2)
                cv2.circle(img_col, (cx[i], cy[i]), 7, (255, 255, 255), -1)
                # cv2.putText(img_col, "a: %.0f" % area[j], (cX[i]-20, cY[i]-20), font, 1, (255, 255, 255), 2)
                # cv2.putText(img_col, "c: %.2f" % circularity[j], (cX[i]-20, cY[i]-50), font, 1, (255, 255, 255), 2)
                cv2.putText(img_col, "i: %.2f" % inertia[j], (cx[i] - 20, cy[i] - 20), font, 1, (255, 255, 255), 2)
                j = j + 1

    print("Filtered number of contours = " + str(j))
    processed_filename = path + image_name + "_contour_filtered.png"
    cv2.imwrite(processed_filename, img_col)

    # PART TWO - DECISION ALGORITHM TODO
    iMaxArea = np.argmax(area)
    return cx[iMaxArea], cy[iMaxArea]


def isThereLife(im):
    """
    Function uses a blob detector to find green objects larger than some specified area
    :param im: bgr image
    :return: Coordinate pixels of green object if detected, None if not
    """

    # Converts images from BGR to HSV
    im_plant = im.copy()
    plant_filename = path + image_name + "_plantblob.png"
    hsv = cv2.cvtColor(im_plant, cv2.COLOR_BGR2HSV)

    # Green mask -> BGR -> GREYSCALE
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    imask = mask > 0
    green = np.zeros_like(im_plant, np.uint8)
    green[imask] = im_plant[imask]
    green_bgr = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
    im_grey = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2GRAY)

    # Detect plant via blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minThreshold = 0
    params.maxThreshold = 200
    params.minArea = 500
    params.maxArea = 100000
    params.minCircularity = 0
    params.minConvexity = 0
    params.minInertiaRatio = 0
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im_grey)
    if len(keypoints) == 0:
        return None
    else:
        im_with_keypoints = cv2.drawKeypoints(im_grey, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(plant_filename, im_with_keypoints)
        return keypoints[0].pt


def getCannyFromBGR(im_bgr):
    """
    Function takes a BGR image, converts to greyscale, applies a Gaussian blur,
    Otsu thresholding and the Canny algorithm for edge-detection

    :param im_bgr: Color image
    :return: Canny image
    """

    # Convert BGR image to greyscale
    im_grey = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filtering (5x5 kernel), then perform Otsu thresholding
    blur = cv2.GaussianBlur(im_grey, (5, 5), 0)
    ret, im_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_filename = path + image_name + "_thresh.png"
    cv2.imwrite(processed_filename, im_thresh)

    # Find edges using Canny algorithm on the thresholded image
    im_canny = cv2.Canny(im_thresh, 0, 255, L2gradient=True)

    # Save and return Canny image
    processed_filename = path + image_name + "_canny.png"
    cv2.imwrite(processed_filename, im_canny)
    return im_canny


def getImage():
    """
    Function takes an image from the RealSense camera
    :return: filename
    """
    # Configure depth and color streams
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Get colour and IR data
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_white_balance, True)
    color_sensor.set_option(rs.option.enable_auto_exposure, True)
    ir_sensor = profile.get_device().first_depth_sensor()
    ir_sensor.set_option(rs.option.emitter_enabled, False)

    for x in range(100):
        pipeline.wait_for_frames()
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    ir_frame = frame.get_infrared_frame(1)
    depth_frame = frame.get_depth_frame()
    if not color_frame or not ir_frame:
        return
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Save the images
    path = 'images/' + timestr
    filename_col = path + "_col.png"
    filename_ir = path + "_ir.png"
    filename_depth = path + "_depth.png"
    filename_colormap = path + "_colormapdepth.png"
    cv2.imwrite(filename_col, color_image)
    cv2.imwrite(filename_ir, ir_image)
    cv2.imwrite(filename_depth, depth_image)
    cv2.imwrite(filename_colormap, depth_colormap)
    return filename_col

