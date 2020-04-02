# encoding: utf-8
import cv2 as cv
import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def cv_show(img,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
#
# # 设置终止条件
# crireria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.001)
#
# # 做一些3D点
# objp = np.zeros((6 * 7, 3), np.float32)
# objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# print(objp)
# objpoints = []
# imgpoints = []
#
# images = glob.glob(r'F:\PyCharm\Camera calibration\*.jpg')
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
#
#     if ret == True:
#         # 画出亚像素精度角点
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (7, 6), (-1, -1), crireria)
#         imgpoints.append(corners2)
#         cv.drawChessboardCorners(img, (7, 6), corners2, ret)
#
#         # 标定
#         # Size imageSize, 在计算相机内部参数和畸变矩阵需要
#         # cameraMatrix 为内部参数矩阵， 输入一个cvMat
#         # distCoeffs为畸变矩阵.
#         # 我们要使用的函数是 cv.calibrateCamera()。它会返回摄像机矩阵，畸变系数，旋转和变换向量等。
#         # mtx内参矩阵， dist畸变系数, rvecs旋转向量，外餐 tvecs平移向量，内参
#         ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)
#
#         print(fname, "rvecs", rvecs)
#         cv.imshow('img', img)
#
#         k = cv.waitKey(500) & 0xff
#         h, w = img.shape[:2]
#
#         # 畸变矫正
#         # 如果缩放系数 alpha = 0，返回的非畸变图像会带有最少量的不想要的像素。
#         # 它甚至有可能在图像角点去除一些像素。如果 alpha = 1，所有的像素都会被返回，
#         # 还有一些黑图像。它还会返回一个 ROI 图像，我们可以用来对结果进行裁剪。
#         newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # roi只是一个元组
#         x, y, w, h = roi
#         print(roi)
#
#         # 去除畸变
#         if k == ord('q'):
#             # undistort
#             dst = cv.undistort(img, mtx, dist, newCameraMatrix=newcameramtx)
#             dst = dst[y:y + h, x:x + w]
#             cv.imshow('undistortimg', dst)
#         else:
#             # remapping。先找到畸变到非畸变的映射方程，再用重映射方程
#             mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
#             dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
#             # dst = dst[y:y+h, x:x+w] #你可以试试注释这一步
#             cv.imshow('undistortimg ', dst)
#             cv.waitKey()
#
#         # 我们可以用反向投影对我们找到的参数的准确性评估，结果约接近0越好
#         # 有了内部参数：畸变参数和旋转变换矩阵，我们可以用cv.projectPoints()去
#         # 把对象点转换到图像点，然后就可以计算变换得到图像与角点检测算法的绝对差了。
#         # 然后我们计算所有标定图像的误差平均值。
#         mean_error = 0
#         for i in range(len(objpoints)):
#             imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#             error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
#             mean_error += error
#         print("total error: ", mean_error / len(objpoints))
#
#
#
#
#
#

# ###又一标定代码###
# # 标定图像保存路径
# photo_path = "F:\PyCharm\Camera calibration"
#
#
# # 标定图像
# def calibration_photo(photo_path):
#     # 设置要标定的角点个数
#     x_nums = 7  # x方向上的角点个数
#     y_nums = 7
#     # 设置(生成)标定图在世界坐标中的坐标
#     world_point = np.zeros((x_nums * y_nums, 3), np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
#     world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)  # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
#     # world_point[:, :2] = 1
#     # .T矩阵的转置
#     # reshape()重新规划矩阵，但不改变矩阵元素
#     # 保存角点坐标
#     world_position = []
#     image_position = []
#     # 设置角点查找限制
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     # 获取所有标定图
#     images = glob.glob(photo_path + '\\*.jpg')
#     # print(images)
#     for image_path in images:
#         image = cv.imread(image_path)
#         gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#         # 查找角点
#         ok, corners = cv.findChessboardCorners(gray, (x_nums, y_nums), None)
#
#         if ok:
#             # 把每一幅图像的世界坐标放到world_position中
#             world_position.append(world_point)
#             # 获取更精确的角点位置
#             exact_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             # 把获取的角点坐标放到image_position中
#             image_position.append(exact_corners)
#             # 可视化角点
#             # image = cv.drawChessboardCorners(image,(x_nums,y_nums),exact_corners,ok)
#             # cv.imshow('image_corner',image)
#             # cv.waitKey(5000)
#     # 计算内参数
#     ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_position, image_position, gray.shape[::-1], None, None)
#     # 将内参保存起来
#     np.savez('C:\\Users\\wlx\\Documents\\py_study\\camera calibration\\data\\intrinsic_parameters', mtx=mtx, dist=dist)
#     print(mtx, dist)
#     # 计算偏差
#     mean_error = 0
#     for i in range(len(world_position)):
#         image_position2, _ = cv.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)
#         error = cv.norm(image_position[i], image_position2, cv.NORM_L2) / len(image_position2)
#         mean_error += error
#     print("total error: ", mean_error / len(image_position))
#
#
# if __name__ == '__main__':
#     calibration_photo(photo_path)


####可以用得上的标定代码####
# #
help(cv.getOptimalNewCameraMatrix)







