import pyrealsense2 as rs
import numpy as np
import cv2
import time


def getimage():

    # colour_sensor = profile.get_device().query_sensors()[1]
    # colour_sensor.set_option(rs.option.enable_auto_exposure, False)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'RS_HS_' + timestr + '.raw'

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, False)
    color_sensor.set_option(rs.option.enable_auto_white_balance, False)
    color_sensor.set_option(rs.option.exposure, 1000)
    filt = rs.save_single_frameset()

    for x in range(100):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()
    filt.process(frame)
    color_frame = frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    print(color_image.dtype)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(3000)
    color_image = color_image[224:831, 869:1049]
    cv2.imwrite('martiansurface.png', color_image)
    # color_image.astype('uint16').tofile(filename)
