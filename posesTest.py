import socket
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import os

timestr = time.strftime("%Y%m%d-%H%M%S")
scanname = 'RS_HS_' + timestr
os.mkdir(scanname)
os.chdir('./' + scanname)
cwd = os.getcwd()

start_home = 1


def movepose(pose, a, v):
    script = "movej(p[{},{},{},{},{},{}], a={}, v={})"
    script = script.format(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], a, v)
    s.send(bytes(script, 'utf-8') + bytes("\n", 'utf-8'))
    time.sleep(0.5)


def takePicture(i, j):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    color_image = color_image[224:831, 869:1049]
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(5)

    filename = "scan_" + str(i) + "_" + str(j) + ".png"
    cv2.imwrite(filename, color_image)
    time.sleep(0.025)


HOST = "129.78.214.100"  # The remote host
PORT = 30003  # The same port as used by the server
print("Starting Program\n")
count = 0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("Socket connected!\n")
if start_home == 1:
    home = np.array([0.6, 0.4, 0.25, 0, 0, 45 * 3.14 / 180])
    movepose(home, 0.5, 0.5)

input("Confirm HS attachment off, then press any key to continue...")
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, False)
color_sensor.set_option(rs.option.enable_auto_white_balance, False)
color_sensor.set_option(rs.option.exposure, 3000)

centralPose = np.array([0.4, 0.4, 0.25, 0, 0, 135 * 3.14 / 180])
movepose(centralPose, 0.5, 0.5)
print("Moving to central pose.\n")
time.sleep(6)
vfov = np.deg2rad(42.5)
hfov = np.deg2rad(69.4)
# hfov = np.deg2rad(42.5)

offset = np.array([-35e-3, 70e-3, 0, 0, 0, 0])

frameCentre = centralPose - offset
pc = rs.pointcloud()
points = rs.points()
colorizer = rs.colorizer()
frames = pipeline.wait_for_frames()
colorized = colorizer.process(frames)
ply = rs.save_to_ply("pointCloud.ply")

ply.set_option(rs.save_to_ply.option_ply_binary, False)
ply.set_option(rs.save_to_ply.option_ply_normals, True)
ply.process(colorized)

depth_frame = frames.get_depth_frame()
depth_image = np.asanyarray(depth_frame.get_data())
images = depth_image
np.save('depth_image', depth_image)
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RealSense', images)
cv2.waitKey(5)

filename = "scan_depth.png"
cv2.imwrite(filename, depth_image)

print("Install HS attachment NOW!\n")
input("Press any key to continue...")
time.sleep(2)

vScanRes = 30e-3
maxDist = 0.085
hWidth = 1280
hScanRes = maxDist * hfov / hWidth

# phiRange = np.array([-vfov/2, 0, vfov/2])

phiRange = np.array([0])
thetaRange = np.arange(-hfov / 2, hfov / 2, hScanRes / maxDist)

yRange = frameCentre[2] + offset[1] * np.tan(phiRange)

xRange = frameCentre[0] + offset[1] * np.tan(thetaRange)

poses = np.zeros((xRange.size, yRange.size, centralPose.size))
j = 0
noFrames = xRange.size * yRange.size

firstPose = np.append(np.array([xRange[0], frameCentre[1], yRange[0]]),
                      np.array([centralPose[3], centralPose[4], centralPose[5] - thetaRange[0]]))
movepose(firstPose, 1, 1)
print("Scan beginning!")
time.sleep(5)

tic = time.perf_counter()
for y in yRange:
    i = 0
    for x in xRange:
        poses[i, j, :] = np.append(np.array([x, frameCentre[1], y]),
                                   np.array([centralPose[3], centralPose[4], centralPose[5] - thetaRange[i]]))

        pose = poses[i, j, :]
        print("Frame: ", i, " of ", noFrames, " @ ", pose)
        movepose(pose, 1, 1)
        takePicture(i, j)

        i = i + 1
    j = j + 1

toc = time.perf_counter()
data = s.recv(1024)
s.close()
pipeline.stop()
print("Scan completed in: ", toc - tic, "seconds.")
print("Program finish")