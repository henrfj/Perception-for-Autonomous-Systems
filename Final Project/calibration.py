import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

rerun_camera_matrix = True

##################### LEFT IMAGES #########################
if rerun_camera_matrix == True:   
    # Implement the number of vertical and horizontal corners
    nb_vertical = 9
    nb_horizontal = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.

    images = glob.glob('Exercises/Final Project/calibration/left-*.png')
    assert images

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Implement findChessboardCorners here
        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints_left.append(corners)

    # get the camera matrix
    ret, mtx_left, dist_left, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_left, gray.shape[::-1], None, None)
    print(ret, mtx_left, dist_left, rvecs, tvecs)
    img_left = cv2.imread('Exercises/Final Project/calibration/left-0000.png') # just to get dimensions
    h,  w = img.shape[:2]
    K_left, roi = cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),0,(5*w,5*h))

    # save images into folder
    i = 0
    for fname in images:
        # undistort
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx_left, dist_left, None, K_left)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        # save image
        cv2.imwrite('Exercises/Final Project/undistorted/left'+str(i)+'.png',dst)
        i+=1
    
    ##################### RIGHT IMAGES #########################

    # Implement the number of vertical and horizontal corners
    nb_vertical = 9
    nb_horizontal = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_right = [] # 2d points in image plane.

    images = glob.glob('Exercises/Final Project/calibration/right-*.png')
    assert images

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Implement findChessboardCorners here
        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints_right.append(corners)

    # get the camera matrix
    ret, mtx_right, dist_right, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_right, gray.shape[::-1], None, None)
    img_right = cv2.imread('Exercises/Final Project/calibration/right-0000.png') # just to get dimensions
    h,  w = img.shape[:2]
    K_right, roi_right = cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),1,(w,h))

    # save images into folder
    i = 0
    for fname in images:
        # undistort
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx_right, dist_right, None, K_right)

        # crop the image
        x,y,w,h = roi_right
        dst = dst[y:y+h, x:x+w]

        # save image
        cv2.imwrite('Exercises/Final Project/undistorted/right'+str(i)+'.png',dst)
        i+=1