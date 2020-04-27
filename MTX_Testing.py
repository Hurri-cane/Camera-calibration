import cv2 as cv
import  numpy as np

def cv_show(img,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

chessboard_size = (15,13)

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


axis = np.float32([[90, 0, 0], [0, 90, 0], [0, 0, -90]]).reshape(-1, 3)  # 坐标轴

img = cv.imread(r'F:\PyCharm\Camera_calibration_GIT\class3\01.jpg')
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
    cv_show(img)