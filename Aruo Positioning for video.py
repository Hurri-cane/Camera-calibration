#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import cv2.aruco as aruco
import  numpy as np


def cv_show(frame,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

aruco_size = 0.076

with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

index = 0
cap = cv.VideoCapture(0,cv.CAP_DSHOW)  #更改API设置
flag = cap.isOpened()
cap.set(3, 1280)
cap.set(4, 720)

Camear_Point = None
rvec = None
tvec = None
point1 = None
point2 = None
while (flag):
    ret, frame = cap.read()
    frame_copy = frame.copy()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedframePoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters)
    font = cv.FONT_HERSHEY_SIMPLEX

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, aruco_size, mtx, dist)  # aruco码边长为0.028m
        (rvec - tvec).any()

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], aruco_size)
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 179, 255))
        cv.putText(frame, "Id: " + str(ids.T), (50, 80), font, 2, (0, 255, 0), 2, cv.LINE_AA)
    ###计算位置

        for j in range(len(rvec)):
            dst, _ = cv.Rodrigues(rvec[j])
            dst = np.mat(dst)
            tr = tvec[j]
            tr = np.mat(tr)
            Camear_Point = -dst.I * tr.T
            Camear_Point = Camear_Point * 1000  # 单位转换为mm
            Xc = int(Camear_Point[0])
            Yc = int(Camear_Point[1])
            Zc = int(Camear_Point[2])
            print('\n', "-------------------------")
            print("Id: " + str(ids[j]) + '坐标为：')
            print('Xc=', Xc, 'mm', 'Yc=', Yc, 'mm', 'Zc=', Zc, 'mm')
            location = tuple(corners[j][0][0])
            # b = (200,200)
            # print(a)
            cv.putText(frame, "Id: " + str(ids[j]) + str([Xc,Yc,Zc]), location, font, 1, (0, 255, 0), 2, cv.LINE_AA)


            #距离计算（仅含有两个标识）
            if j == 0:
                point1 = np.mat([Xc ,Yc ,Zc])
            if j == 1:
                point2 = np.mat([Xc ,Yc ,Zc])
                distance = np.sqrt((point1-point2)*((point1-point2).T))
                distance = int(distance)
                cv.putText(frame, "Distance: "+str(distance), (50, 200), font, 1, (0, 0, 255), 2, cv.LINE_AA)



    else:
        cv.putText(frame, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Capture_Paizhao", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\class5\0" + str(index) + ".jpg", frame_copy)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv.destroyAllWindows()





