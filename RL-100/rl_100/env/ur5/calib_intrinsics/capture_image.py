import os
import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
index = 0

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    cv2.imshow("RealSense", cv2.resize(color_image, (1280, 720)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f'{os.path.dirname(os.path.abspath(__file__))}/saved_images/image_{index}.png'
        while os.path.exists(filename):
            index += 1
            filename = f'{os.path.dirname(os.path.abspath(__file__))}/saved_images/image_{index}.png'
        cv2.imwrite(filename, color_image)
        print(f"Saved image as {filename}")
        index += 1
    elif key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
