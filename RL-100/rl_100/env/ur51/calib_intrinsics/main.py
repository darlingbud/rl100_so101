import sys, os, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((14, 10), 0.02, 0.015, aruco_dict)

save_board = False
if save_board:
    # check if the board is correct
    image = board.generateImage((2560, 2560))

    # save the image to "aruco_calib.jpg"
    cv2.imwrite("aruco_calib.jpg", image)

    plt.figure()
    plt.imshow(image, cmap='gray')
    #plt.title('DICT_4X4_250 6x5 ChAruco pattern')
    plt.axis('off')
    plt.show()


parameters =  cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
detector = cv2.aruco.CharucoDetector(board, detectorParams=parameters)

all_obj_pts = []
all_img_pts = []
all_ids = []

for i in sorted(glob(f'{os.path.dirname(os.path.abspath(__file__))}/saved_images_depth/image_*.png')):
    frame = cv2.imread(i)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c_corners, c_ids, corners, ids = detector.detectBoard(gray)
    ret = len(c_corners)
    print(f'{i}: found {ret} corners')
    if ret > 0:
        objPoints, imgPoints = board.matchImagePoints(c_corners, c_ids)
        all_obj_pts.append(objPoints)
        all_img_pts.append(imgPoints)
        all_ids.append(c_ids)

    imsize = (gray.shape[1], gray.shape[0])

all_obj_pts = np.array(all_obj_pts)
all_img_pts = np.array(all_img_pts)

print(all_obj_pts.shape, all_img_pts.shape)


# Remove cv2.CALIB_RATIONAL_MODEL for non-fisheye lens
# ret, K, d, rvec, tvec = cv2.calibrateCamera(all_obj_pts, all_img_pts, imsize, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL)
ret, K, d, rvec, tvec = cv2.calibrateCamera(all_obj_pts, all_img_pts, imsize, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

np.set_printoptions(suppress=True)
print("Image size = ", imsize)
print("Reprojection error = ", ret)
print("Intrinsic parameter K =\n", K)
print("Distortion parameters d = (k1, k2, p1, p2, k3, k4, k5, k6) =\n", d)

for i in range(len(d[0])):
    if np.abs(d[0][i]) > 1:
        print(f'[WARN] distCoeffs[{i}] is too large: {d[0][i]}.  Possibly an overfitting.')

assert ret < 1.0
