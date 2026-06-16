import time
import numpy as np
import pyrealsense2 as rs
from realsense import RealSense


camera = RealSense()
camera.start()

while True:
    t = time.time()
    frame = camera.get_frame()
    print("Hz:", 1 / (time.time() - t))
