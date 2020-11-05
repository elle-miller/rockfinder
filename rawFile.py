import pyrealsense2 as rs
import numpy as np
import cv2
import time


def getimage():
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
    #green = color_image()
    ir_image = np.asanyarray(ir_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Save the images
    path = '/images/' + timestr
    filename_col = path + "_col.png"
    filename_ir = path + "_ir.png"
    filename_depth = path + "_depth.png"
    filename_colormap = path + "_colormapdepth.png"
    cv2.imwrite(filename_col, color_image)
    cv2.imwrite(filename_ir, ir_image)
    cv2.imwrite(filename_depth, depth_image)
    cv2.imwrite(filename_colormap, depth_colormap)

    return 0
