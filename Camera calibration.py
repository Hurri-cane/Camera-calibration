#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import  numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import degrees as dg


def cv_show(img,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


Path1 = 'F:\PyCharm\Camera_calibration_GIT\class1'



# 定义棋盘大小: 注意此处是内部的行、列角点个数，不包含最外边两列，否则会出错
chessboard_size = (15,13)
a = np.prod(chessboard_size)
# 生成195×3的矩阵，用来保存棋盘图中15*13个内角点的3D坐标，也就是物体点坐标
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
# 通过np.mgrid生成对象的xy坐标点，每个棋盘格大小是18mm
# 最终得到z=0的objp为(0,0,0), (1*13,0,0), (2*13,0,0) ,...
objp[:, :2] = np.mgrid [0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * 18

# print("object is %f", objp)

# 定义数组，来保存监测到的点
obj_points = []  # 保存世界坐标系的三维点
img_points = []  # 保存图片坐标系的二维点

# 设置终止条件： 迭代30次或者变动 < 0.001
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 读取目录下的所有图片
calibration_paths = glob.glob(Path1+'\*.jpg')

useful_img = 0
# 为方便显示使用tqdm显示进度条
for image_path in tqdm(calibration_paths):
    # 读取图片
    img = cv.imread(image_path)
    # x,y = img.shape[:2]
    # ratio = y/x
    # img = cv.resize(img, (int(750*ratio),750))
    # 图像二值化
    gray = cv.cvtColor(img, cv. COLOR_BGR2GRAY)
    # cv_show(gray)
    # 找到棋盘格内角点位置
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:
        obj_points.append(objp)
        # 亚像素级角点检测，在角点检测中精确化角点位置
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img_points.append(corners2)

        # 在图中标注角点,方便查看结果
        img = cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        # img = cv.resize(img, (400,600))
        useful_img+=1
        # cv_show(img)

print("finish all the pic count, useful image amount to ",useful_img)


# 相机标定
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape, None, None)
# 其中fx ＝ f/dX ,fy = f/dY ,分别称为x轴和y轴上的归一化焦距
#u0和v0则表示的是光学中心，即摄像机光轴与图像平面的交点，通常位于图像中心处，故其值常取分辨率的一半。

# 显示和保存参数
print("#######相机内参#######")
print(mtx)
print("#######畸变系数#######")
print(dist)
print("#######相机旋转矩阵#######")
print(rvecs)
print("#######相机平移矩阵#######")
print(tvecs)
np.savez(Path1+'\class_mtx.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs) #分别使用mtx,dist,rvecs,tvecs命名数组

# mtx_mat = np.mat(mtx)
# mtx_mat_T = mtx_mat.I
# #定义像素坐标系中的点
# point1_uv = np.mat([20,30,1])
# point1_xy = np.dot(mtx_mat_T,point1_uv. T)
# print(point1_xy)


# --------------------------------------------------------
# 使用一张图片看看去畸变之后的效果
img2 = cv.imread(Path1+r'\028.jpg')
# img2 = cv.resize(img2, (int(750 * ratio), 750))
cv_show(img2)
print("orgininal img_point  array shape",img2.shape)
# img2.shape[:2]取图片 高、宽；
h,  w = img2.shape[:2]
print("pic's hight, weight: %f,  %f"%(h, w))
# img2.shape[:3]取图片的 高、宽、通道
# h,  w ,n= img2.shape[:3]
# print("PIC shape", (h, w, n))


newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h),1, (w, h))  # 自由比例参数

dst = cv.undistort(img2, mtx, dist, None, newCameraMtx)

# 根据前面ROI区域裁剪图片
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv_show(dst)
cv.imwrite(r'F:\PyCharm\Camera_calibration_GIT\Camera calibration\Calibresult5.jpg', dst)



# --------------------------------------------------------
# 计算所有图片的平均重投影误差
total_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(img_points[i], img_points2, cv.NORM_L2)/len(img_points2)
    total_error += error
print("total error: {}".format(total_error/len(obj_points)))




# --------------------------------------------------------
# 加载相机标定的内参数、外参数矩阵
with np.load(Path1+r'\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

print("#######加载相机内参和畸变矩阵#######")
print(mtx, dist)



# --------------------------------------------------------
# # 定义棋盘大小
chessboard_size = (15,13)

# 世界坐标系下的物体位置矩阵（Z=0）
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * 18

# 像素坐标
test_img = cv.imread(Path1+r"\026.jpg")
gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
# cv_show(test_img)
# 找到图像平面点角点坐标
ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

if ret:
    _, R, T, _, = cv.solvePnPRansac(objp, corners, mtx, dist)
    print("旋转向量", R)
    print("平移向量", T)

sita_x = dg(R[0][0])
sita_y = dg(R[1][0])
sita_z = dg(R[2][0])

print("sita_x is  ", sita_x,'度')
print("sita_y is  ", sita_y,'度')
print("sita_z is  ", sita_z,'度')

# --------------------------------------------------------






# --------------------------------------------------------



# --------------------------------------------------------

# # 加载相机标定的数据
# with np.load(r'F:\PyCharm\Camera calibration\class3\class3.npz') as X:
#     mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, corners, imgpts):
    """
    在图片上画出三维坐标轴
    :param img: 图片原数据
    :param corners: 图像平面点坐标点
    :param imgpts: 三维点投影到二维图像平面上的坐标
    :return:
    """
    # corners[0]是图像坐标系的坐标原点；imgpts[0]-imgpts[3] 即3D世界的坐标系点投影在2D世界上的坐标
    corner = tuple(corners[0].ravel())
    # 沿着3个方向分别画3条线
    cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 2)
    cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 2)
    cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 2)
    return img

# #定义棋盘大小
# chessboard_size = (15,13)

# 初始化目标坐标系的3D点
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)*18

# 初始化三维坐标系
axis = np.float32([[90, 0, 0], [0, 90, 0], [0, 0, -90]]).reshape(-1, 3)  # 坐标轴

# 加载打包所有图片数据
images = glob.glob(Path1+r'\026.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv_show(img)
    # 找到图像平面点坐标点
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # PnP计算得出旋转向量和平移向量
        _, rvecs, tvecs, _ = cv.solvePnPRansac(objp, corners, mtx, dist)
        print("旋转变量", rvecs)
        print("平移变量", tvecs)
        # 计算三维点投影到二维图像平面上的坐标
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # 把坐标显示图片上
        img = draw(img, corners, imgpts)
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\3d_2d_project\3d_2d_project5.jpg",img)
        cv_show(img)


# cv.destroyAllWindows()

