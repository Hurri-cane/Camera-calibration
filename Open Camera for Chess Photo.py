
#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import  numpy as np


# --------------------------------------------------------
# 打开摄像头
cap = cv.VideoCapture(1,cv.CAP_DSHOW)  #更改API设置
flag = cap.isOpened()
cap.set(3, 1280)
cap.set(4, 720)


# --------------------------------------------------------
# 定义棋盘
chessboard_size = (15,13)
a = np.prod(chessboard_size)
# 生成195×3的矩阵，用来保存棋盘图中15*13个内角点的3D坐标，也就是物体点坐标
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
# 通过np.mgrid生成对象的xy坐标点，每个棋盘格大小是18mm
# 最终得到z=0的objp为(0,0,0), (1*13,0,0), (2*13,0,0) ,...
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * 18

# 设置终止条件： 迭代3次或者变动 < 0.1
# criteria = None
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3, 0.1)

obj_points = []  # 保存世界坐标系的三维点
img_points = []  # 保存图片坐标系的二维点




# --------------------------------------------------------
# 检测照片

index = 0
while (flag):
    ret1, frame = cap.read()
    frame_save = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret2, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret2 == True:
        obj_points.append(objp)
        # 亚像素级角点检测，在角点检测中精确化角点位置
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img_points.append(corners2)

        # 在图中标注角点,方便查看结果
        frame = cv.drawChessboardCorners(frame, chessboard_size, corners2, ret2)
        # img = cv.resize(img, (400,600))


    # frame = cv.flip(frame, 1)  # 水平翻转
    cv.imshow("Capture_Paizhao", frame)
    # print(cap.get(3),cap.get(4))
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\class3\0" + str(index) + ".jpg", frame_save)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv.destroyAllWindows()
